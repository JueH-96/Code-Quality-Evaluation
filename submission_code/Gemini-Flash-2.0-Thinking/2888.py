from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return -1

        # Find the dominant element of the entire array
        counts = Counter(nums)
        dominant_element = -1
        for num, count in counts.items():
            if count * 2 > n:
                dominant_element = num
                break

        if dominant_element == -1:
            return -1 # Should not happen based on problem description

        for i in range(n - 1):
            left_array = nums[:i + 1]
            right_array = nums[i + 1:]

            n_left = len(left_array)
            n_right = len(right_array)

            if n_left > 0:
                left_counts = Counter(left_array)
                left_dominant = -1
                if dominant_element in left_counts and left_counts[dominant_element] * 2 > n_left:
                    left_dominant = dominant_element
            else:
                left_dominant = -1

            if n_right > 0:
                right_counts = Counter(right_array)
                right_dominant = -1
                if dominant_element in right_counts and right_counts[dominant_element] * 2 > n_right:
                    right_dominant = dominant_element
            else:
                right_dominant = -1

            if left_dominant == dominant_element and right_dominant == dominant_element:
                return i

        return -1