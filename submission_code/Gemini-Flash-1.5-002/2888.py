class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        dominant_element = None
        max_freq = 0
        for num, freq in counts.items():
            if freq * 2 > len(nums):
                dominant_element = num
                max_freq = freq
                break

        if dominant_element is None:
            return -1

        for i in range(len(nums) -1):
            left_counts = {}
            for num in nums[:i+1]:
                left_counts[num] = left_counts.get(num,0) + 1
            
            right_counts = {}
            for num in nums[i+1:]:
                right_counts[num] = right_counts.get(num,0) + 1

            left_dominant = None
            left_max_freq = 0
            for num, freq in left_counts.items():
                if freq * 2 > len(nums[:i+1]):
                    left_dominant = num
                    left_max_freq = freq
                    break

            right_dominant = None
            right_max_freq = 0
            for num, freq in right_counts.items():
                if freq * 2 > len(nums[i+1:]):
                    right_dominant = num
                    right_max_freq = freq
                    break

            if left_dominant == dominant_element and right_dominant == dominant_element:
                return i
        return -1