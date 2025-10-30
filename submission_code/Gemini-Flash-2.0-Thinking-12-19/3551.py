class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def calculate_xor_score_fast(arr):
            length = len(arr)
            if length == 0:
                return 0
            if length % 2 == 0:
                return arr[0] ^ arr[-1] if length >= 2 else arr[0] if length == 1 else 0
            else:
                xor_sum = 0
                for x in arr:
                    xor_sum ^= x
                return xor_sum

        results = []
        for l_i, r_i in queries:
            max_xor_score = 0
            for i in range(l_i, r_i + 1):
                for j in range(i, r_i + 1):
                    subarray = nums[i:j+1]
                    xor_score = calculate_xor_score_fast(subarray)
                    max_xor_score = max(max_xor_score, xor_score)
            results.append(max_xor_score)
        return results