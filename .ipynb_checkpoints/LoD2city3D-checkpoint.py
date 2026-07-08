# -*- coding: utf-8 -*-
# env/geo3D_sim04
#########################
# helper functions to create LoD1 3D City Model from volunteered public data (OpenStreetMap) with elevation via a raster DEM.

# author: arkriger - 2023 - 2026
# github: https://github.com/AdrianKriger/geo3D

# script credit:
#    - building height from osm building:level: https://github.com/ualsg/hdb3d-code/blob/master/hdb2d.py - Filip Biljecki <filip@nus.edu.sg>
#    - extruder: https://github.com/cityjson/misc-example-code/blob/master/extruder/extruder.py - Hugo Ledoux <h.ledoux@tudelft.nl>

# additional thanks:
#    - cityjson community: https://github.com/cityjson
#########################


import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def extract_lod2_surfaces_to_gdf(buildings, vertices, crs="EPSG:4326"):
    """
    Parses a dictionary of CityJSON CityObjects to extract individual explicit 
    LoD2 semantic surfaces (Roofs, Walls, Grounds), performs 3D vector calculations 
    for physical attributes, and builds a clean GeoDataFrame.

    Parameters:
    -----------
    buildings : dict
        The dictionary of city objects (typically acquired from cityjson.reader.CityJSON.cityobjects)
    vertices : list
        The list of total vertex coordinates from the cityjson object registry.
    crs : str, optional
        The coordinate reference system to assign to the output footprint layer. Default is "EPSG:4326".

    Returns:
    --------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the semantic surface components, true 3D mathematical attributes,
        and flat projected 2D footprint geometries.
    """
    lod2_records = []

    for b_id, building in buildings.items():
        # Store building-level attributes
        b_attributes = building.attributes if building.attributes else {}

        # Iterate over available geometries (a building can have multiple LoDs)
        for geom in building.geometry:
            # We target LoD2 representations exclusively
            if str(geom.lod) != '2' and geom.lod != 2:
                continue

            # cjio parses geometry surfaces into an internal registry mapping to boundaries
            # geom.surfaces typically maps face indices to semantic types
            if not hasattr(geom, 'surfaces') or geom.surfaces is None:
                continue

            for srf_idx, srf_info in geom.surfaces.items():
                surface_type = srf_info.get('type', 'Unknown')  # e.g., 'RoofSurface', 'WallSurface'
                surface_indices = srf_info.get('surface_idx', [])

                # Extract the actual 3D polygon loops belonging to this specific semantic surface
                for idx_tuple in surface_indices:
                    # In CityJSON, boundaries can be nested arrays depending on geometry type (Solid vs MultiSurface)
                    # We can retrieve the vertex indices for the outer ring of this polygon face
                    try:
                        if geom.type.lower() == 'solid':
                            # idx_tuple looks like [shell_idx, face_idx]
                            face_vertices_indices = geom.boundaries[idx_tuple[0]][idx_tuple[1]][0]
                        else:
                            # MultiSurface/CompositeSurface: idx_tuple looks like [face_idx]
                            face_vertices_indices = geom.boundaries[idx_tuple[0]][0]
                    except (IndexError, TypeError):
                        continue

                    # Fetch the actual coordinates [[x1, y1, z1], [x2, y2, z2], ...]
                    face_coords = [vertices[v_idx] for v_idx in face_vertices_indices]
                    if len(face_coords) < 3:
                        continue

                    # Close the polygon ring if it isn't closed
                    if face_coords[0] != face_coords[-1]:
                        face_coords.append(face_coords[0])
                        
                    pts = np.array(face_coords)

                    # 3. Perform 3D Vector Mathematics for Spatial Analysis
                    # Calculate normal vector of the 3D surface using two edge vectors
                    v1 = pts[1] - pts[0]
                    v2 = pts[2] - pts[0]
                    normal = np.cross(v1, v2)
                    norm_len = np.linalg.norm(normal)
                    if norm_len == 0:
                        continue
                    normal = normal / norm_len

                    # True 3D Surface Area (using the magnitude of the cross product)
                    flat_polygon_3d = Polygon(pts)
                    true_3d_area = flat_polygon_3d.area

                    # Calculate Pitch / Tilt (angle relative to the horizontal ground plane [0, 0, 1])
                    cos_tilt = abs(normal[2])
                    tilt_degrees = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))

                    # Calculate Orientation / Aspect (azimuth direction the surface faces)
                    aspect_degrees = np.degrees(np.arctan2(normal[0], normal[1])) % 360

                    # 4. Generate a 2D Footprint for GIS / GeoPandas compatibility
                    # Project the 3D coordinates down to the X-Y plane
                    xy_footprint = Polygon([(p[0], p[1]) for p in face_coords])

                    # Collect the compiled data entry
                    record = {
                        'building_id': b_id,
                        'surface_type': surface_type,
                        'geometry_type': geom.type,
                        'true_3d_area': true_3d_area,
                        'tilt_deg': tilt_degrees,
                        'aspect_deg': aspect_degrees,
                        'mean_z': np.mean(pts[:, '2' if pts.ndim > 1 else 2]),
                        'geometry': xy_footprint
                    }

                    # Unpack original building attributes safely
                    for k, v in b_attributes.items():
                        record[f'attr_{k}'] = v
                        
                    lod2_records.append(record)

    # Convert the collected records directly into a spatial GeoDataFrame
    if lod2_records:
        gdf = gpd.GeoDataFrame(lod2_records, geometry='geometry', crs=crs)
    else:
        gdf = gpd.GeoDataFrame(columns=['building_id', 'surface_type', 'geometry'], geometry='geometry', crs=crs)
        
    return gdf