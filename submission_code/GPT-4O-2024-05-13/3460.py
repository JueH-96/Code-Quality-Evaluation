from typing import List
from math import comb

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        def count_inversions(perm):
            inv_count = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if perm[i] > perm[j]:
                        inv_count[j] += 1
            return inv_count
        
        def valid_permutation(perm):
            inv_count = count_inversions(perm)
            for end_i, cnt_i in requirements:
                if sum(inv_count[:end_i + 1]) != cnt_i:
                    return False
            return True
        
        def generate_permutations(arr, l, r):
            if l == r:
                if valid_permutation(arr):
                    return 1
                return 0
            count = 0
            for i in range(l, r + 1):
                arr[l], arr[i] = arr[i], arr[l]
                count = (count + generate_permutations(arr, l + 1, r)) % MOD
                arr[l], arr[i] = arr[i], arr[l]
            return count
        
        arr = list(range(n))
        return generate_permutations(arr, 0, n - 1)