class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        count = 0

        # Case 1: No students are selected (k = 0)
        happy_with_zero_selected = True
        for num in nums:
            if 0 >= num:
                happy_with_zero_selected = False
                break
        if happy_with_zero_selected:
            count += 1

        # Case 2: Some students are selected (0 < k < n)
        for k in range(1, n):
            # Check if selecting the first k students makes everyone happy
            possible = True
            # Selected students
            for i in range(k):
                if k <= nums[i]:
                    possible = False
                    break
            if not possible:
                continue

            # Not selected students
            for i in range(k, n):
                if k >= nums[i]:
                    possible = False
                    break

            if possible:
                count += 1

        # Case 3: All students are selected (k = n)
        happy_with_n_selected = True
        for num in nums:
            if n <= num:
                happy_with_n_selected = False
                break
        if happy_with_n_selected:
            count += 1

        return count