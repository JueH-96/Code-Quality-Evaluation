class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        counts = {}
        left = 0
        total = 0
        min_val = nums[0]
        max_val = nums[0]
        
        for right, value in enumerate(nums):
            counts[value] = counts.get(value, 0) + 1
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value
            while max_val - min_val > 2:
                counts[nums[left]] -= 1
                if counts[nums[left]] == 0:
                    del counts[nums[left]]
                    if nums[left] == min_val or nums[left] == max_val:
                        if counts:
                            min_val = min(counts.keys())
                            max_val = max(counts.keys())
                        else:
                            min_val = nums[right]
                            max_val = nums[right]
                left += 1
            total += right - left + 1
        return total