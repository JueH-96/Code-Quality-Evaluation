class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        n = len(nums)
        sign_len = n - 1
        if m > sign_len:
            return 0
        
        # Create the sign array
        sign = []
        for i in range(n - 1):
            diff = nums[i+1] - nums[i]
            if diff > 0:
                sign.append(1)
            elif diff < 0:
                sign.append(-1)
            else:
                sign.append(0)
        
        count = 0
        # Check each possible subarray of length m in the sign array
        for i in range(len(sign) - m + 1):
            if sign[i:i+m] == pattern:
                count += 1
        
        return count