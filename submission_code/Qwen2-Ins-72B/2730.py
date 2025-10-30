class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] | num)
        
        suffix = [0]
        for num in reversed(nums):
            suffix.append(suffix[-1] | num)
        suffix = list(reversed(suffix))
        
        ans = 0
        for i, num in enumerate(nums):
            left, right = prefix[i], suffix[i+1]
            ans = max(ans, left | (num * (2 ** k)) | right)
            
        return ans