class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0

        max_sum = 0

        # Calculate for the first window
        current_sum = 0
        window_counts = {}
        distinct_count = 0
        for i in range(k):
            current_sum += nums[i]
            if nums[i] not in window_counts:
                window_counts[nums[i]] = 0
            window_counts[nums[i]] += 1
            if window_counts[nums[i]] == 1:
                distinct_count += 1

        if distinct_count >= m:
            max_sum = current_sum

        # Slide the window
        for i in range(k, n):
            left_element = nums[i - k]
            right_element = nums[i]

            # Update sum
            current_sum -= left_element
            current_sum += right_element

            # Update counts for the element leaving the window
            window_counts[left_element] -= 1
            if window_counts[left_element] == 0:
                distinct_count -= 1

            # Update counts for the element entering the window
            if right_element not in window_counts:
                window_counts[right_element] = 0
            window_counts[right_element] += 1
            if window_counts[right_element] == 1:
                distinct_count += 1

            # Check if the current window is almost unique
            if distinct_count >= m:
                max_sum = max(max_sum, current_sum)

        return max_sum