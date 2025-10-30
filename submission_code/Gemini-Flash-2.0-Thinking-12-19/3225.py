class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq_map = {}
        left = 0
        max_len = 0
        for right in range(len(nums)):
            num = nums[right]
            freq_map[num] = freq_map.get(num, 0) + 1
            while freq_map[num] > k:
                left_num = nums[left]
                freq_map[left_num] -= 1
                if freq_map[left_num] == 0:
                    del freq_map[left_num]
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len