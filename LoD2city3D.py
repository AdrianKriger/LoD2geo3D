# -*- coding: utf-8 -*-
# env/geo3D_sim04
#########################
# helper functions to support LoD2geo3D. An extension and compliment to geo3D.

# author: arkriger - 2026
# github: https://github.com/AdrianKriger/LoD2geo3D
#########################

import numpy as np
from shapely.geometry import Polygon

def extract_lod_surfaces(buildings, vertices, crs_name="EPSG:4326"):
    """
    Parses a dictionary of CityJSON CityObjects to extract individual explicit 
    LoD1 and LoD2 surfaces, performs 3D vector calculations for physical attributes, 
    and builds a standard Python dictionary dataset.
    """
    extracted_features = []

    for b_id, building in buildings.items():
        b_attributes = getattr(building, 'attributes', {}) or building.get('attributes', {})
        geometries = getattr(building, 'geometry', []) if hasattr(building, 'geometry') else building.get('geometry', [])

        for geom in geometries:
            geom_type = getattr(geom, 'type', geom.get('type', '')).lower()
            boundaries = getattr(geom, 'boundaries', geom.get('boundaries', []))
            lod_val = str(getattr(geom, 'lod', geom.get('lod', '')))
            
            # Container to temporarily hold faces we want to process for this geometry
            faces_to_process = []

            # --- CASE 1: HANDLE LoD1 ---
            if lod_val.startswith('1'):
                # LoD1 lacks semantics; we manually flatten and traverse all boundary faces
                try:
                    if geom_type == 'solid':
                        # Loops through shells, then faces
                        for shell in boundaries:
                            for face in shell:
                                faces_to_process.append(('GenericSurface', face[0]))
                    else:
                        # MultiSurface / CompositeSurface
                        for face in boundaries:
                            faces_to_process.append(('GenericSurface', face[0]))
                except (IndexError, TypeError):
                    continue

            # --- CASE 2: HANDLE LoD2 ---
            elif lod_val.startswith('2'):
                surfaces = getattr(geom, 'surfaces', None) or geom.get('semantics', {}).get('surfaces', [])
                if not surfaces:
                    continue
                
                iterator = surfaces.items() if hasattr(surfaces, 'items') else enumerate(surfaces)
                for srf_idx, srf_info in iterator:
                    surface_type = srf_info.get('type', 'Unknown')
                    surface_indices = srf_info.get('surface_idx', [])

                    for idx_tuple in surface_indices:
                        try:
                            if geom_type == 'solid':
                                face = boundaries[idx_tuple[0]][idx_tuple[1]][0]
                            else:
                                face = boundaries[idx_tuple[0]][0]
                            faces_to_process.append((surface_type, face))
                        except (IndexError, TypeError):
                            continue
            else:
                # Skip any LoD0 or LoD3+ if present
                continue

            # --- COMMON 3D MATHEMATICS & EXPORT PIPELINE ---
            for surface_type, face_vertices_indices in faces_to_process:
                face_coords = [vertices[v_idx] for v_idx in face_vertices_indices]
                if len(face_coords) < 3:
                    continue

                if face_coords[0] != face_coords[-1]:
                    unique_pts = np.array(face_coords)
                    face_coords.append(face_coords[0])
                else:
                    unique_pts = np.array(face_coords[:-1])

                pts = np.array(face_coords)

                # 1. Normal Vector
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len == 0:
                    continue
                unit_normal = normal / norm_len

                # 2. Newell's Method for True 3D Area
                raw_area_vector = np.zeros(3)
                for i in range(len(unique_pts)):
                    p0 = unique_pts[i]
                    p1 = unique_pts[(i + 1) % len(unique_pts)]
                    raw_area_vector += np.cross(p0, p1)
                true_3d_area = 0.5 * np.linalg.norm(raw_area_vector)

                # 3. Tilt & Aspect
                cos_tilt = abs(unit_normal[2])
                tilt_degrees = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
                aspect_degrees = np.degrees(np.arctan2(unit_normal[0], unit_normal[1])) % 360

                # 4. Save Properties & Geometry Payload
                xy_footprint_coords = [[float(p[0]), float(p[1])] for p in face_coords]

                properties = {
                    'building_id': b_id,
                    'lod': lod_val,
                    'surface_type': surface_type,
                    'geometry_type': geom_type,
                    'true_3d_area': float(true_3d_area),
                    'tilt_deg': float(tilt_degrees),
                    'aspect_deg': float(aspect_degrees),
                    'mean_z': float(np.mean(pts[:, 2]))
                }

                for k, v in b_attributes.items():
                    properties[f'attr_{k}'] = v

                extracted_features.append({
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [xy_footprint_coords]
                    }
                })

    return {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': crs_name}},
        'features': extracted_features
    }