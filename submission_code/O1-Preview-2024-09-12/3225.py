class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        freq = defaultdict(int)
        left = 0
        max_len = 0
        badCount = 0
        for right in range(len(nums)):
            freq[nums[right]] +=1
            if freq[nums[right]] == k+1:
                badCount +=1
            while badCount > 0:
                freq[nums[left]] -=1
                if freq[nums[left]] == k:
                    badCount -=1
                left +=1
            max_len = max(max_len, right - left +1)
        return max_len