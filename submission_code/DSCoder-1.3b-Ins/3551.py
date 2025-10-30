class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] ^ num)
        
        ans = []
        for l, r in queries:
            max_xor = max(prefix[r+1], prefix[l] ^ 0)
            ans.append(max_xor)
        
        return ans