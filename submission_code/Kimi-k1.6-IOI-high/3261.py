class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        # Compute the initial OR of all elements
        for num in nums:
            ans |= num
        
        # Check each bit from highest (bit 29) to lowest (bit 0)
        for bit in reversed(range(30)):
            # Tentatively turn off this bit
            candidate = ans & ~(1 << bit)
            # Count how many numbers have this bit turned off
            mask = 1 << bit
            count = 0
            for num in nums:
                if (num & mask) == 0:
                    count += 1
            # Check if we can turn off this bit
            if count >= (n - k):
                ans = candidate
        
        return ans