class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total = sum(nums)
        # For any partition, left sum = sum(nums[0...i])
        # Right sum = total - left sum
        # The difference is left - right = 2 * left - total.
        # Because 2 * left is always even, the parity of the difference is determined solely by total.
        # If total is even, then 2 * left - total is even for every possible left sum.
        # If total is odd, then 2 * left - total is odd for every choice of left sum.
        # Therefore, the count of valid partitions is (n-1) if total is even, or 0 if total is odd.
        if total % 2 != 0:
            return 0
        return len(nums) - 1