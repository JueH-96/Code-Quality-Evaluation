class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        dominant_element = -1
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        for num, count in counts.items():
            if count * 2 > n:
                dominant_element = num
                break

        if dominant_element == -1:
            return -1

        for i in range(n - 1):
            left_arr = nums[:i+1]
            right_arr = nums[i+1:]

            left_dominant = -1
            left_counts = {}
            for num in left_arr:
                left_counts[num] = left_counts.get(num, 0) + 1
            for num, count in left_counts.items():
                if count * 2 > len(left_arr):
                    left_dominant = num
                    break

            right_dominant = -1
            right_counts = {}
            for num in right_arr:
                right_counts[num] = right_counts.get(num, 0) + 1
            for num, count in right_counts.items():
                if count * 2 > len(right_arr):
                    right_dominant = num
                    break
            
            if left_dominant == dominant_element and right_dominant == dominant_element:
                return i
        return -1