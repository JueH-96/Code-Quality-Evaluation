class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        positions = []
        n = len(nums)
        
        # Collect positions of '1's
        for i in range(n):
            if nums[i] == 1:
                positions.append(i)
        
        m = len(positions)
        
        # If there are no '1's, return 0
        if m == 0:
            return 0
        
        # If there is only one '1', only one way to split
        if m == 1:
            return 1 % MOD
        
        # Calculate the product of distances between consecutive '1's
        answer = 1
        for i in range(1, m):
            distance = positions[i] - positions[i-1]
            answer = (answer * distance) % MOD
        
        return answer