from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # If there's only one element, there's exactly one way to partition.
        if n == 1:
            return 1
        
        # 1) Collect for each distinct value the first and last occurrence in nums.
        positions = {}
        for i, val in enumerate(nums):
            if val not in positions:
                positions[val] = [i, i]
            else:
                positions[val][1] = i
        
        # 2) We'll use a difference array "diff" to mark which boundaries are forbidden.
        #    diff[i] will track increments/decrements for coverage of the boundary between i and i+1.
        diff = [0] * (n - 1)  # We have n-1 possible boundaries in a list of length n.
        
        # 3) For each distinct number x, it occupies an interval [start, end].
        #    We must forbid placing boundaries from start to end-1.
        for start, end in positions.values():
            # If the interval has positive length, mark it in diff.
            if start <= end - 1:
                diff[start] += 1  # increment coverage starting at "start"
                if end <= n - 2:  # end is actually (end_of_interval + 1) in diff, if within range
                    diff[end] -= 1
        
        # 4) Build the "coverage" array from diff via prefix sums:
        #    coverage[i] > 0 means the boundary i is forbidden (it lies in at least one interval).
        coverage = [0] * (n - 1)
        coverage[0] = diff[0]
        for i in range(1, n - 1):
            coverage[i] = coverage[i - 1] + diff[i]
        
        # 5) Count how many boundaries are "free" (coverage == 0). 
        #    Each free boundary can either be cut or not, independently → 2^(count_of_free_boundaries).
        free_boundaries = sum(1 for val in coverage if val == 0)
        
        # 6) Compute the result with fast exponentiation mod 10^9+7.
        return pow(2, free_boundaries, MOD)