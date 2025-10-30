from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        MAX_VAL = 100000
        min_idx = [-1] * (MAX_VAL + 1)
        
        for j, d in enumerate(elements):
            if d <= MAX_VAL:
                if min_idx[d] == -1 or j < min_idx[d]:
                    min_idx[d] = j
        
        divisors_list = [[] for _ in range(MAX_VAL + 1)]
        for i in range(1, MAX_VAL + 1):
            j = i
            while j <= MAX_VAL:
                divisors_list[j].append(i)
                j += i
        
        ans_for_value = [-1] * (MAX_VAL + 1)
        for v in range(1, MAX_VAL + 1):
            best_index = float('inf')
            for d in divisors_list[v]:
                if min_idx[d] != -1:
                    if min_idx[d] < best_index:
                        best_index = min_idx[d]
            if best_index != float('inf'):
                ans_for_value[v] = best_index
        
        return [ans_for_value[g] for g in groups]