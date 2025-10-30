from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        target = nums.copy()
        possible = [set() for _ in range(n)]
        for j in range(n):
            possible[j].add(0)
        
        for k in range(len(queries)):
            l, r, val = queries[k]
            temp_updates = {}
            # Compute new possible sets for each j in the query's range
            for j in range(n):
                if l <= j <= r:
                    current_s = possible[j]
                    new_s = set()
                    for s in current_s:
                        new_s.add(s)
                        new_val = s + val
                        if new_val <= target[j]:
                            new_s.add(new_val)
                    temp_updates[j] = new_s
            # Update the possible sets
            for j in temp_updates:
                possible[j] = temp_updates[j]
            
            # Check if all targets are met
            all_met = True
            for j in range(n):
                if target[j] not in possible[j]:
                    all_met = False
                    break
            if all_met:
                return k + 1  # since k is 0-based index
        
        return -1