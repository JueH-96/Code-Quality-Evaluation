class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ways = 0
        for k in range(n + 1):
            if k == 0:
                # All students are not selected
                # Check if for all students, k < nums[i]
                if all(k < num for num in nums):
                    ways += 1
            elif k == n:
                # All students are selected
                # Check if for all students, k > nums[i]
                if all(k > num for num in nums):
                    ways += 1
            else:
                # Check if for all students, if selected, k > nums[i], else k < nums[i]
                # Since the array is sorted, the first k students are selected
                # So for the first k students, k > nums[i]
                # For the remaining n - k students, k < nums[i]
                if k > nums[k - 1] and k < nums[k]:
                    ways += 1
        return ways