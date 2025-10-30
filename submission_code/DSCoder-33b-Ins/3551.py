class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        prefix_xor = [0]
        for num in nums:
            prefix_xor.append(prefix_xor[-1] ^ num)

        ans = []
        for l, r in queries:
            xor = prefix_xor[r + 1] ^ prefix_xor[l]
            max_xor = xor
            for i in range(l, r + 1):
                max_xor = max(max_xor, xor ^ nums[i])
            ans.append(max_xor)

        return ans