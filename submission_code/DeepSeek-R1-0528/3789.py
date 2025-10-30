import sys
from collections import defaultdict

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        total_subarrays = n * (n + 1) // 2
        m = len(conflictingPairs)
        graph = defaultdict(list)
        for a, b in conflictingPairs:
            a_idx = a - 1
            b_idx = b - 1
            if a_idx > b_idx:
                a_idx, b_idx = b_idx, a_idx
            graph[b_idx].append(a_idx)
        
        max_left = [-1] * n
        for i in range(n):
            if i in graph:
                max_left[i] = max(graph[i])
        
        left = 0
        independent_count = 0
        for right in range(n):
            if max_left[right] != -1:
                left = max(left, max_left[right] + 1)
            independent_count += (right - left + 1)
        U = total_subarrays - independent_count
        
        edges = []
        for a, b in conflictingPairs:
            a_idx = a - 1
            b_idx = b - 1
            if a_idx > b_idx:
                a_idx, b_idx = b_idx, a_idx
            edges.append((a_idx, b_idx))
        
        by_x = defaultdict(list)
        for a_idx, b_idx in edges:
            by_x[a_idx].append(b_idx)
        
        min_y_global = [10**9] * (n + 1)
        min_y_global[n] = 10**9
        for i in range(n - 1, -1, -1):
            min_y_here = 10**9
            if i in by_x:
                min_y_here = min(by_x[i])
            min_y_global[i] = min(min_y_global[i + 1], min_y_here)
        
        min_y_global2 = [10**9] * (n + 1)
        for i in range(n - 1, -1, -1):
            min_val = 10**9
            if i in by_x:
                for y in by_x[i]:
                    if min_val > y:
                        min_val = y
            min_y_global2[i] = min(min_y_global2[i + 1], min_val)
        
        min_y_global = min_y_global2
        
        ans = 0
        for a_idx, b_idx in edges:
            a_val = a_idx + 1
            b_val = b_idx + 1
            total_contain_e = a_val * (n - b_val + 1)
            area_covered = 0
            min_y_arr = [10**9] * (a_val + 2)
            min_y_arr[a_val + 1] = 10**9
            for L in range(a_val, 0, -1):
                L_idx = L - 1
                min_y_arr[L] = min_y_arr[L + 1]
                if L_idx in by_x:
                    for y in by_x[L_idx]:
                        if y >= b_idx:
                            if min_y_arr[L] > y:
                                min_y_arr[L] = y
                if min_y_arr[L] == 10**9:
                    continue
                R_low = max(b_val, min_y_arr[L] + 1)
                if R_low > n:
                    coverage = 0
                else:
                    coverage = n - R_low + 1
                area_covered += coverage
            X_e = total_contain_e - area_covered
            res_e = (total_subarrays - U) + X_e
            if res_e > ans:
                ans = res_e
        return ans