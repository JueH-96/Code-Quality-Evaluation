class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()

        # Initialize the count of ways
        ways = 0

        # Iterate through each possible number of selected students
        for i in range(n + 1):
            if i == 0:
                # If no students are selected, all students will be happy if their nums[i] is less than 0
                if all(nums[j] < 0 for j in range(n)):
                    ways += 1
            elif i == n:
                # If all students are selected, all students will be happy if their nums[i] is greater than n
                if all(nums[j] > n for j in range(n)):
                    ways += 1
            else:
                # Check if the current number of selected students satisfies the conditions
                if all(nums[j] < i for j in range(i)) and all(nums[j] > i for j in range(i, n)):
                    ways += 1

        return ways