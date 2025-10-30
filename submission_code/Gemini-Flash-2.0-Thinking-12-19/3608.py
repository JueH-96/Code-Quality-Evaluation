import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        gcd_to_index_sets = {}
        for i in range(1 << n):
            current_indices = []
            current_subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    current_indices.append(j)
                    current_subsequence.append(nums[j])
            if not current_subsequence:
                continue
            current_gcd = current_subsequence[0]
            for k in range(1, len(current_subsequence)):
                current_gcd = math.gcd(current_gcd, current_subsequence[k])
            index_set = tuple(sorted(current_indices))
            if current_gcd not in gcd_to_index_sets:
                gcd_to_index_sets[current_gcd] = []
            gcd_to_index_sets[current_gcd].append(set(index_set))
            
        count = 0
        mod = 10**9 + 7
        for g in gcd_to_index_sets:
            index_sets_g = gcd_to_index_sets[g]
            unique_index_sets_g = list(set(tuple(sorted(s)) for s in index_sets_g))
            set_list = [set(s) for s in unique_index_sets_g]
            num_sets = len(set_list)
            for i in range(num_sets):
                for j in range(num_sets):
                    if set_list[i].isdisjoint(set_list[j]):
                        count = (count + 1) % mod
                        
        return count