# -*- coding: utf-8 -*-
# env/geo3D_sim04
#########################
# helper functions to support LoD2geo3D. An extension and compliment to geo3D.

# author: arkriger - 2026
# github: https://github.com/AdrianKriger/LoD2geo3D
#########################

from typing import Optional, Any, Union

import numpy as np
import pandas as pd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, LinearRing, MultiPolygon, MultiLineString, MultiPoint, shape, mapping

from pyproj import CRS, Transformer 

class GeoDataFrameLite(pd.DataFrame):
    """A lightweight GeoDataFrame-like wrapper with .crs support."""

    _metadata = ["_crs"]

    @property
    def _constructor(self):
        return GeoDataFrameLite

    @property
    def crs(self) -> Optional[CRS]:
        """Return the CRS object, or None if unset."""
        return getattr(self, "_crs", None)

    @crs.setter
    def crs(self, crs_input: Any):
        """Set CRS from user input (EPSG, WKT, PROJ string, CRS object)."""
        self._crs = CRS.from_user_input(crs_input)

    def to_json(self, indent: int = None) -> str:
        """Serialize to GeoJSON FeatureCollection."""
        features = []
        for _, row in self.iterrows():
            geom = row.get("geometry")
            props = {k: v for k, v in row.items() if k != "geometry"}
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": mapping(geom) if geom is not None else None
            })
        fc = {"type": "FeatureCollection", "features": features}
        return json.dumps(fc, indent=indent)

    @classmethod
    def from_json(cls, json_input: Union[str, dict]) -> "GeoDataFrameLite":
        """Read GeoJSON string or dict into GeoDataFrameLite."""
        if isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input

        if data.get("type") != "FeatureCollection":
            raise ValueError("Expected GeoJSON FeatureCollection")

        rows = []
        for feat in data["features"]:
            geom = shape(feat["geometry"]) if feat.get("geometry") else None
            props = feat.get("properties", {})
            props["geometry"] = geom
            rows.append(props)

        df = cls(rows)
        return df

    def estimate_utm_crs(self, datum_name: str = "WGS 84") -> CRS:
        """
        Estimate the best UTM CRS for the current geometries using pyproj.database.query_utm_crs_info.
        Works like GeoPandas. Returns a pyproj CRS object.
        """
        if "geometry" not in self.columns or self.empty:
            return None

        # Compute combined bounding box
        bounds = [g.bounds for g in self["geometry"] if g is not None]
        if not bounds:
            return None
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)

        # Build AreaOfInterest for pyproj query
        aoi = AreaOfInterest(
            west_lon_degree=minx,
            south_lat_degree=miny,
            east_lon_degree=maxx,
            north_lat_degree=maxy,
        )

        # Query UTM CRS info
        utm_crs_list = query_utm_crs_info(datum_name=datum_name, area_of_interest=aoi)
        if not utm_crs_list:
            raise ValueError("No suitable UTM CRS found for the bounding box.")

        # Return pyproj CRS object of the first recommended UTM
        return CRS.from_epsg(utm_crs_list[0].code)

    def to_crs(self, crs_input: Any) -> "GeoDataFrameLite":
        """
        Reproject all geometries to a new CRS.
        Returns a new GeoDataFrameLite with transformed geometries.
        """
        if "geometry" not in self.columns or self.empty:
            return self.copy()

        if self.crs is None:
            raise ValueError("Current CRS is not set. Set df.crs before calling to_crs().")

        new_crs = CRS.from_user_input(crs_input)
        transformer = Transformer.from_crs(self.crs, new_crs, always_xy=True)

        def _reproject(geom):
            if geom is None:
                return None
            return transform(transformer.transform, geom)

        df = self.copy()
        df["geometry"] = df["geometry"].apply(_reproject)
        df.crs = new_crs
        return df

def extract_lod_surfaces(buildings, vertices, transform_meta=None, crs_name="EPSG:32734"):
    """
    Parses CityJSON 2.0 CityObjects, resolves LoD1/LoD2 semantics by cross-referencing
    the 'values' shell mapping array, and returns a real-world scaled FeatureCollection.
    """
    extracted_features = []
    
    # Extract scale and translation factors to output true world coordinates
    scale = transform_meta.get('scale', [1.0, 1.0, 1.0]) if transform_meta else [1.0, 1.0, 1.0]
    translate = transform_meta.get('translate', [0.0, 0.0, 0.0]) if transform_meta else [0.0, 0.0, 0.0]

    for b_id, building in buildings.items():
        b_attributes = building.get('attributes', {})
        geometries = building.get('geometry', [])

        for geom in geometries:
            geom_type = geom.get('type', '').lower()
            boundaries = geom.get('boundaries', [])
            lod_val = str(geom.get('lod', ''))
            
            faces_to_process = []

            # --- CASE 1: HANDLE LoD1 ---
            if lod_val.startswith('1'):
                if geom_type == 'solid':
                    for shell in boundaries:
                        for face in shell:
                            faces_to_process.append(('GenericSurface', face[0]))
                else:
                    for face in boundaries:
                        faces_to_process.append(('GenericSurface', face[0]))

            # --- CASE 2: HANDLE LoD2 (CityJSON 2.0 Schema Fixed) ---
            elif lod_val.startswith('2'):
                semantics = geom.get('semantics', {})
                surfaces_templates = semantics.get('surfaces', [])
                semantic_values = semantics.get('values', [])

                if not surfaces_templates or not semantic_values:
                    continue

                if geom_type == 'solid':
                    # Loop through shells and faces using standard indexing positions
                    for shell_idx, shell in enumerate(boundaries):
                        for face_idx, face in enumerate(shell):
                            try:
                                # Look up surface template index out of the nested values shell mapping
                                sem_idx = semantic_values[shell_idx][face_idx]
                                if sem_idx is not None:
                                    surface_type = surfaces_templates[sem_idx].get('type', 'Unknown')
                                else:
                                    surface_type = 'Unclassified'
                                # CityJSON Solid geometry faces are wrapped in an outer list ring: [ [v1, v2, v3] ]
                                faces_to_process.append((surface_type, face[0]))
                            except (IndexError, TypeError):
                                continue
                else:
                    # MultiSurface / CompositeSurface tracking
                    for face_idx, face in enumerate(boundaries):
                        try:
                            sem_idx = semantic_values[face_idx]
                            if sem_idx is not None:
                                surface_type = surfaces_templates[sem_idx].get('type', 'Unknown')
                            else:
                                surface_type = 'Unclassified'
                            faces_to_process.append((surface_type, face[0]))
                        except (IndexError, TypeError):
                            continue
            else:
                continue

            # --- 3D VECTOR MATH & TRANSFORM PIPELINE ---
            for surface_type, face_vertices_indices in faces_to_process:
                # 1. Transform raw integer vertices back into real-world coordinates immediately
                face_coords = []
                for v_idx in face_vertices_indices:
                    raw_v = vertices[v_idx]
                    real_x = (raw_v[0] * scale[0]) + translate[0]
                    real_y = (raw_v[1] * scale[1]) + translate[1]
                    real_z = (raw_v[2] * scale[2]) + translate[2]
                    face_coords.append([real_x, real_y, real_z])
                
                if len(face_coords) < 3:
                    continue

                # Ensure polygon ring closure
                if face_coords[0] != face_coords[-1]:
                    unique_pts = np.array(face_coords)
                    face_coords.append(face_coords[0])
                else:
                    unique_pts = np.array(face_coords[:-1])

                pts = np.array(face_coords)

                # 2. Compute true real-world surface normals
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len == 0:
                    continue
                unit_normal = normal / norm_len

                # 3. Newell's Method for true 3D Area (now in square meters!)
                raw_area_vector = np.zeros(3)
                for i in range(len(unique_pts)):
                    p0 = unique_pts[i]
                    p1 = unique_pts[(i + 1) % len(unique_pts)]
                    raw_area_vector += np.cross(p0, p1)
                true_3d_area = 0.5 * np.linalg.norm(raw_area_vector)

                # 4. Tilt & Aspect Angle Orientations
                cos_tilt = abs(unit_normal[2])
                tilt_degrees = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
                aspect_degrees = np.degrees(np.arctan2(unit_normal[0], unit_normal[1])) % 360

                # 5. Extract flat real-world XY footprints for GeoJSON serialization
                #xy_footprint_coords = [[float(p[0]), float(p[1])] for p in face_coords]
                # Inside your surface extraction loop, save the full 3D coordinates:
                xyz_3d_coords = [[float(p[0]), float(p[1]), float(p[2])] for p in face_coords]

                properties = {
                    'building_id': b_id,
                    'lod': lod_val,
                    'surface_type': surface_type,
                    'geometry_type': geom_type,
                    '3d_area': float(true_3d_area),
                    'tilt_deg': float(tilt_degrees),
                    'aspect_deg': float(aspect_degrees),
                    'mean_z_absolute': float(np.mean(pts[:, 2]))
                }

                for k, v in b_attributes.items():
                    properties[f'attr_{k}'] = v

                extracted_features.append({
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [xyz_3d_coords]
                    }
                })

    return {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': crs_name}},
        'features': extracted_features
    }