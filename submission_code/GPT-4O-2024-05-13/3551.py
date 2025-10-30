class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def find_max_xor_subarray(arr):
            max_xor = 0
            n = len(arr)
            for i in range(n):
                current_xor = 0
                for j in range(i, n):
                    current_xor ^= arr[j]
                    max_xor = max(max_xor, current_xor)
            return max_xor
        
        results = []
        for l, r in queries:
            subarray = nums[l:r+1]
            max_xor = find_max_xor_subarray(subarray)
            results.append(max_xor)
        
        return results