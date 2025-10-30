class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        flips = [0] * n
        result = 0
        
        for j in range(n):
            sum_flips = 0
            # Calculate the sum of flips affecting the current position j
            # Flips at j-2, j-1, and j (if they are valid starting positions)
            if (j - 2) >= 0 and (j - 2) <= (n - 3):
                sum_flips += flips[j - 2]
            if (j - 1) >= 0 and (j - 1) <= (n - 3):
                sum_flips += flips[j - 1]
            if j <= (n - 3):
                sum_flips += flips[j]
            
            current = nums[j] ^ (sum_flips % 2)
            if current == 0:
                if j <= (n - 3):
                    flips[j] = 1
                    result += 1
                else:
                    return -1
        
        # Simulate the flips to verify all elements are 1
        simulated = nums.copy()
        for i in range(n):
            if flips[i]:
                for k in range(i, i + 3):
                    if k < n:
                        simulated[k] ^= 1
        
        if all(x == 1 for x in simulated):
            return result
        else:
            return -1