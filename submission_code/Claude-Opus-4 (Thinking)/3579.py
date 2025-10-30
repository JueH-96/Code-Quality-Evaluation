class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        from itertools import permutations
        
        return max(int(''.join(bin(num)[2:] for num in perm), 2) 
                   for perm in permutations(nums))