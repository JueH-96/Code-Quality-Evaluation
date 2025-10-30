import collections
import heapq
import bisect

class Solution:
    def checkValidCuts(self, n: int, rectangles: list[list[int]]) -> bool:
        
        def solve(coords_list: list[tuple[int, int]], N_dim: int) -> bool:
            if not coords_list:
                return False

            L = len(coords_list)
            
            all_coords_set = set()
            for s, e in coords_list:
                all_coords_set.add(s)
                all_coords_set.add(e)
            
            V_orig = sorted(list(all_coords_set))
            k_orig = len(V_orig)
            
            if k_orig == 0: # No coordinates, no cuts possible to form sections with rects
                return False

            rank_map_orig = {val: i for i, val in enumerate(V_orig)}

            # 1. Validity of cuts: valid_cut[j] is true if V_orig[j] is a valid cut.
            # A cut Yc is valid if no rectangle (s,e) has s < Yc < e.
            valid_cut = [True] * k_orig
            
            # Difference array for point coverage: V_orig[j] is crossed if s < V_orig[j] < e.
            # mark_invalid_delta[p] is sum of changes at rank p.
            # If rect (s,e) has ranks (rank_s, rank_e), points V_orig[rank_s+1]...V_orig[rank_e-1] are crossed.
            mark_invalid_delta = collections.defaultdict(int)
            for s_coord, e_coord in coords_list:
                rank_s = rank_map_orig[s_coord]
                rank_e = rank_map_orig[e_coord]
                # Check if there are any points in V_orig strictly between s_coord and e_coord
                if rank_s + 1 <= rank_e - 1: 
                    mark_invalid_delta[rank_s + 1] += 1
                    # The decrement applies after the last crossed point V_orig[rank_e-1].
                    # So it should be at rank_e.
                    mark_invalid_delta[rank_e] -= 1
            
            current_mark = 0
            for j in range(k_orig):
                current_mark += mark_invalid_delta[j]
                if current_mark > 0:
                    valid_cut[j] = False

            # 2. is_S1_ok[j] = exists rect R_p with e_p <= V_orig[j]
            is_S1_ok = [False] * k_orig
            s1_val_markers = [False] * k_orig # True if some e_p == V_orig[j]
            for _, e_coord in coords_list:
                # e_coord could be N_dim or 0, ensure it's in rank_map_orig
                if e_coord in rank_map_orig:
                    s1_val_markers[rank_map_orig[e_coord]] = True
            
            if k_orig > 0 and V_orig[0] in rank_map_orig : is_S1_ok[0] = s1_val_markers[0] # check if V_orig[0] exists
            for j in range(1, k_orig):
                is_S1_ok[j] = is_S1_ok[j-1] or s1_val_markers[j]

            # 3. is_S3_ok[j] = exists rect R_p with s_p >= V_orig[j]
            is_S3_ok = [False] * k_orig
            s3_val_markers = [False] * k_orig # True if some s_p == V_orig[j]
            for s_coord, _ in coords_list:
                if s_coord in rank_map_orig:
                     s3_val_markers[rank_map_orig[s_coord]] = True
            
            if k_orig > 0 and V_orig[k_orig-1] in rank_map_orig: is_S3_ok[k_orig-1] = s3_val_markers[k_orig-1]
            for j in range(k_orig-2, -1, -1):
                is_S3_ok[j] = is_S3_ok[j+1] or s3_val_markers[j]

            # 4. min_end_coord_for_S2[j] = min { e_p | s_p >= V_orig[j] }
            min_end_coord_for_S2 = [float('inf')] * k_orig
            rects_by_s_rank = [[] for _ in range(k_orig)]
            for s_coord, e_coord in coords_list:
                if s_coord in rank_map_orig: # Should always be true as V_orig from these
                    rects_by_s_rank[rank_map_orig[s_coord]].append(e_coord)
            
            min_heap = [] # Stores actual e_coord values
            for j in range(k_orig - 1, -1, -1):
                for e_val in rects_by_s_rank[j]:
                    heapq.heappush(min_heap, e_val)
                if min_heap:
                    min_end_coord_for_S2[j] = min_heap[0]
            
            # 5. exists_valid_S3_at_or_after[j] (is there Y2=V_orig[m] for m >= j that's valid for S3?)
            exists_valid_S3_at_or_after = [False] * (k_orig + 1) 
            for j in range(k_orig - 1, -1, -1):
                s3_possible_here = False
                if V_orig[j] < N_dim: # Y2 must be < N_dim
                    s3_possible_here = valid_cut[j] and is_S3_ok[j]
                exists_valid_S3_at_or_after[j] = s3_possible_here or exists_valid_S3_at_or_after[j+1]
            
            # 6. Main loop: Iterate Y1 = V_orig[j]
            for j in range(k_orig):
                if V_orig[j] <= 0 or V_orig[j] >= N_dim: # Y1 must be in (0, N_dim)
                    continue
                
                if not valid_cut[j] or not is_S1_ok[j]:
                    continue

                E_min_for_S2 = min_end_coord_for_S2[j] 
                if E_min_for_S2 == float('inf'): 
                    continue

                m_lower = bisect.bisect_left(V_orig, E_min_for_S2) 
                m_actual_lower = max(j + 1, m_lower)

                if m_actual_lower >= k_orig: 
                    continue
                
                if exists_valid_S3_at_or_after[m_actual_lower]:
                    return True
            
            return False

        coords_y = []
        coords_x = []
        for r_sx, r_sy, r_ex, r_ey in rectangles:
            coords_y.append((r_sy, r_ey))
            coords_x.append((r_sx, r_ex))
        
        if solve(coords_y, n):
            return True
        # For vertical cuts, N_dim is also n (grid width)
        if solve(coords_x, n): 
            return True
            
        return False