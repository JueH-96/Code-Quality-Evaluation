class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        def max_xor_subarray(nums):
            max_xor = 0
            current_xor = 0
            prefix_xor = defaultdict(int)
            prefix_xor[0] = -1
            for i, num in enumerate(nums):
                current_xor ^= num
                max_xor = max(max_xor, current_xor ^ prefix_xor[current_xor ^ 0])
                if current_xor not in prefix_xor:
                    prefix_xor[current_xor] = i
            return max_xor
        
        result = []
        for l, r in queries:
            result.append(max_xor_subarray(nums[l:r+1]))
        return result