class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Compute the entire array's AND
        if not nums:
            return 0
        total_and = nums[0]
        for num in nums[1:]:
            total_and &= num
            if total_and == 0:
                break  # Further ANDing won't change the result
        
        if total_and != 0:
            return 1
        
        # Now, find the maximum number of subarrays with AND 0
        prev = set()
        count = 0
        for num in nums:
            curr = set()
            curr.add(num)
            if prev:
                for p in prev:
                    curr.add(p & num)
            if 0 in curr:
                count += 1
                prev = set()  # Reset for the next subarray
            else:
                prev = curr.copy()
        return count