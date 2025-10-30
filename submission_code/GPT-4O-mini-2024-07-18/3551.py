class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        from itertools import accumulate
        
        # Precompute the prefix XOR array
        prefix_xor = [0] + list(accumulate(nums, lambda x, y: x ^ y))
        
        # Function to get the maximum XOR score of any subarray in nums[l..r]
        def max_xor_in_range(l, r):
            max_xor = 0
            for start in range(l, r + 1):
                for end in range(start, r + 1):
                    current_xor = prefix_xor[end + 1] ^ prefix_xor[start]
                    max_xor = max(max_xor, current_xor)
            return max_xor
        
        # Process each query
        answer = []
        for l, r in queries:
            answer.append(max_xor_in_range(l, r))
        
        return answer