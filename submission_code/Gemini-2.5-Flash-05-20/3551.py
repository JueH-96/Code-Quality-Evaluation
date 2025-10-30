import sys

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        N = len(nums)

        # 1. Calculate prefix_xor array
        # pxor[i] stores XOR sum of nums[0]...nums[i-1]
        # pxor[0] is 0
        pxor = [0] * (N + 1)
        for i in range(N):
            pxor[i+1] = pxor[i] ^ nums[i]

        # 2. Calculate current_subarray_score[k][p] for 0 <= k <= p < N
        # This table stores the XOR score of the specific subarray nums[k..p]
        # The values nums[i] are non-negative, so 0 is a valid score.
        current_subarray_score = [[0] * N for _ in range(N)] 
        
        for k in range(N):
            for p in range(k, N):
                length = p - k + 1
                if length == 1:
                    # Score of [x0] is x0
                    current_subarray_score[k][p] = nums[k]
                elif length % 2 == 1: # Odd length (>= 3)
                    # Score of [x0, ..., xm-1] with odd length m (>=3) is x0 ^ xm-1
                    current_subarray_score[k][p] = nums[k] ^ nums[p]
                else: # Even length
                    # Score of [x0, ..., xm-1] with even length m is x0 ^ x1 ^ ... ^ xm-1 (XOR sum)
                    current_subarray_score[k][p] = pxor[p+1] ^ pxor[k]
        
        # 3. Calculate max_xor_score_in_range[i][j]
        # This table stores the maximum XOR score among *all* subarrays nums[k..p]
        # such that i <= k <= p <= j.
        # This is a 2D dynamic programming problem.
        max_xor_score_in_range = [[0] * N for _ in range(N)]

        # Base cases: for subarrays of length 1 (i.e., [i, i])
        for i in range(N):
            max_xor_score_in_range[i][i] = current_subarray_score[i][i] # which is nums[i]

        # Fill table for increasing length of query ranges [i, j]
        # Length `L` here refers to the length of the query range (j - i + 1),
        # not the length of the subarray (p - k + 1).
        for L in range(2, N + 1):
            for i in range(N - L + 1):
                j = i + L - 1
                
                # The maximum score for the range nums[i..j] is the maximum of:
                # 1. The score of the entire subarray nums[i..j] itself.
                # 2. The maximum score found in the sub-range nums[i+1..j] (excluding nums[i]).
                # 3. The maximum score found in the sub-range nums[i..j-1] (excluding nums[j]).
                
                max_val = current_subarray_score[i][j]
                
                # Max from sub-range [i+1, j]
                max_val = max(max_val, max_xor_score_in_range[i+1][j])
                
                # Max from sub-range [i, j-1]
                max_val = max(max_val, max_xor_score_in_range[i][j-1])
                
                max_xor_score_in_range[i][j] = max_val

        # 4. Answer queries
        results = []
        for l, r in queries:
            results.append(max_xor_score_in_range[l][r])
        
        return results