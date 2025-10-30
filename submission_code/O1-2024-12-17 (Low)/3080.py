class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # First, compute the AND of all elements.
        full_and = nums[0]
        for x in nums[1:]:
            full_and &= x
        
        # If the AND of the entire array is nonzero, the minimum sum of subarray ANDs
        # can only be achieved by taking the entire array as a single subarray.
        # Hence answer is 1.
        if full_and != 0:
            return 1
        
        # Otherwise (full_and == 0), we can partition the array into multiple subarrays
        # each having AND = 0. We use a greedy approach: build a subarray until its
        # AND becomes 0, then start a new one.
        ans = 0
        current_and = None
        
        for num in nums:
            current_and = num if current_and is None else (current_and & num)
            if current_and == 0:
                ans += 1
                current_and = None
        
        return ans