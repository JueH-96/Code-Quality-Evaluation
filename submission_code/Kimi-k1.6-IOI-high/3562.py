from bisect import bisect_left

class Solution:
    def maximumWeight(self, intervals):
        intervals_with_idx = list(enumerate(intervals))
        sorted_intervals = sorted(intervals_with_idx, key=lambda x: (x[1][1], x[0]))
        
        INF = float('-inf')
        end_pos = [[] for _ in range(5)]
        weights = [[] for _ in range(5)]
        indices = [[] for _ in range(5)]
        prefix_max = [[] for _ in range(5)]
        max_indices = [[] for _ in range(5)]
        
        end_pos[0] = [INF]
        weights[0] = [0]
        indices[0] = [[]]
        prefix_max[0] = [0]
        max_indices[0] = [0]
        
        for (original_idx, (l, r, w)) in sorted_intervals:
            for k in range(4, 0, -1):
                prev_k = k - 1
                prev_end = end_pos[prev_k]
                if not prev_end:
                    continue
                idx = bisect_left(prev_end, l) - 1
                if idx < 0:
                    continue
                current_pmax = prefix_max[prev_k]
                max_prev = current_pmax[idx]
                max_idx = max_indices[prev_k][idx]
                prev_indices_list = indices[prev_k][max_idx]
                new_weight = max_prev + w
                new_indices = prev_indices_list + [original_idx]
                
                current_end = end_pos[k]
                current_weights = weights[k]
                current_indices = indices[k]
                current_pmax_k = prefix_max[k]
                current_max_indices_k = max_indices[k]
                
                current_end.append(r)
                current_weights.append(new_weight)
                current_indices.append(new_indices)
                
                if not current_pmax_k:
                    current_pmax_k.append(new_weight)
                    current_max_indices_k.append(0)
                else:
                    prev_pmax = current_pmax_k[-1]
                    new_pmax = max(prev_pmax, new_weight)
                    current_pmax_k.append(new_pmax)
                    if new_pmax > prev_pmax:
                        current_max_indices_k.append(len(current_weights) - 1)
                    else:
                        current_max_indices_k.append(current_max_indices_k[-1])
        
        candidates = [(0, [])]
        for k in range(1, 5):
            for i in range(len(weights[k])):
                candidates.append((weights[k][i], indices[k][i]))
        
        candidates.sort(key=lambda x: (-x[0], x[1]))
        max_weight = candidates[0][0]
        for weight, idx_list in candidates:
            if weight < max_weight:
                break
            if weight == max_weight:
                return idx_list
        return []