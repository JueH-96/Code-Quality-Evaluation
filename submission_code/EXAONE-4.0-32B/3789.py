import sys
from collections import defaultdict
import heapq

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        total_subarrays = n * (n + 1) // 2
        
        distinct_set = set()
        freq = defaultdict(int)
        for pair in conflictingPairs:
            a, b = pair
            if a > b:
                a, b = b, a
            distinct_set.add((a, b))
            freq[(a, b)] += 1
        
        if not distinct_set:
            return total_subarrays
        
        distinct_cons = set()
        for (a, b) in distinct_set:
            distinct_cons.add((a - 1, b - 1))
        
        min_y_arr = [n] * n
        for (x, y) in distinct_cons:
            if y < min_y_arr[x]:
                min_y_arr[x] = y
        
        g = [n] * n
        if n > 0:
            g[n - 1] = min_y_arr[n - 1]
            for i in range(n - 2, -1, -1):
                g[i] = min(min_y_arr[i], g[i + 1])
        
        total_area = 0
        for i in range(n):
            total_area += (n - g[i])
        base_good = total_subarrays - total_area
        
        sorted_cons = sorted(distinct_cons, key=lambda p: p[0], reverse=True)
        non_dominated = []
        min_y_val = n
        for (x, y) in sorted_cons:
            if y < min_y_val:
                non_dominated.append((x, y))
                min_y_val = y
        
        lists = [[] for _ in range(n)]
        for (x, y) in distinct_cons:
            lists[x].append(y)
        second_min = [n] * n
        for x in range(n):
            if lists[x]:
                lists[x].sort()
                if len(lists[x]) >= 2:
                    second_min[x] = lists[x][1]
                else:
                    second_min[x] = n
        
        best = base_good
        for e in non_dominated:
            x_e, y_e = e
            a_orig = x_e + 1
            b_orig = y_e + 1
            if a_orig > b_orig:
                a_orig, b_orig = b_orig, a_orig
            if freq[(a_orig, b_orig)] > 1:
                candidate_good = base_good
            else:
                backup_min_y = min_y_arr[x_e]
                backup_g = g[:x_e + 1]
                
                min_y_arr[x_e] = second_min[x_e]
                
                if x_e < n - 1:
                    g[x_e] = min(min_y_arr[x_e], g[x_e + 1])
                else:
                    g[x_e] = min_y_arr[x_e]
                for i in range(x_e - 1, -1, -1):
                    g[i] = min(min_y_arr[i], g[i + 1])
                
                diff = 0
                for i in range(x_e + 1):
                    diff += (backup_g[i] - g[i])
                new_area = total_area + diff
                candidate_good = total_subarrays - new_area
                
                for i in range(x_e + 1):
                    g[i] = backup_g[i]
                min_y_arr[x_e] = backup_min_y
            
            if candidate_good > best:
                best = candidate_good
        
        return best