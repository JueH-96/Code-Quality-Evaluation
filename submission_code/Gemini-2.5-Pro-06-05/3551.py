from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)

        # Step 1: Precompute all subarray XOR scores.
        # B[L][i] stores the score of the subarray nums[i...i+L].
        # The length of this subarray is L+1.
        B = [[0] * n for _ in range(n)]
        for i in range(n):
            B[0][i] = nums[i]
        
        for L in range(1, n):
            for i in range(n - L):
                # The XOR score has a recursive structure that can be proven using
                # its relationship with Pascal's triangle modulo 2.
                # score(a_0, ..., a_k) = score(a_0^a_1, ..., a_{k-1}^a_k).
                # This translates to B[L][i] = B[L-1][i] ^ B[L-1][i+1].
                B[L][i] = B[L - 1][i] ^ B[L - 1][i + 1]

        # Step 2 & 3: Use dynamic programming to find the maximum score in any range [l, r].
        # F[l][r] will store the max XOR score of any subarray within nums[l..r].
        F = [[0] * n for _ in range(n)]
        
        # We iterate through all possible ending positions 'r' of the range [l, r].
        for r in range(n):
            # Base case: a single-element range [r, r].
            # The only subarray is nums[r], and its score is nums[r].
            F[r][r] = B[0][r]
            
            # max_ending_at_r tracks the maximum score among subarrays ending at 'r'
            # and starting at or after the current 'l' being considered.
            # This is M_r[l] = max_{k=l..r} score(nums[k..r]).
            max_ending_at_r = B[0][r]

            # Iterate 'l' from r-1 down to 0 to compute F[l][r].
            for l in range(r - 1, -1, -1):
                # The score of the subarray nums[l..r] has length r-l+1.
                # This corresponds to L=r-l, i=l in the B table, so B[r-l][l].
                score_l_r = B[r - l][l]
                
                # Update the max score for subarrays ending at 'r' and starting >= l.
                max_ending_at_r = max(max_ending_at_r, score_l_r)
                
                # The maximum score in nums[l..r] is the maximum of:
                # 1. The maximum score in nums[l..r-1] (which is F[l][r-1]).
                # 2. The maximum score of subarrays starting at or after l and ending at r 
                #    (which is max_ending_at_r, representing M_r[l]).
                F[l][r] = max(F[l][r - 1], max_ending_at_r)

        # Step 4: Answer all queries in O(1) each using the precomputed F table.
        ans = [F[l][r] for l, r in queries]
            
        return ans