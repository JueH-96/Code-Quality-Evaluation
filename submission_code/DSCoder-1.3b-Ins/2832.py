class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        left, right = 0, 0
        counter = collections.Counter()
        max_length = 0
        
        while right < len(nums):
            counter[nums[right]] -= 1
            if counter[nums[right]] == 0:
                counter.pop(nums[right])
            if nums[right] in counter:
                k -= 1
            if k < 0:
                counter[nums[left]] += 1
                if counter[nums[left]] == 1:
                    k += 1
                left += 1
            max_length = max(max_length, right - left + 1)
            right += 1
        
        return max_length