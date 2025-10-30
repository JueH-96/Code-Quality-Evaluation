import collections
from typing import List

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        """
        Calculates the maximum number of non-conflicting subarrays after removing one pair.
        """
        if not conflictingPairs:
            return n * (n + 1) // 2

        partners = collections.defaultdict(list)
        for u, v in conflictingPairs:
            partners[u].append(v)
            partners[v].append(u)

        # M[i] = smallest partner of i greater than i
        # M2[i] = second smallest partner of i greater than i
        M = [n + 1] * (n + 1)
        M2 = [n + 1] * (n + 1)
        
        for i in range(1, n + 1):
            gt_partners = sorted([p for p in partners[i] if p > i])
            if len(gt_partners) > 0:
                M[i] = gt_partners[0]
            if len(gt_partners) > 1:
                M2[i] = gt_partners[1]

        # R[i] = min(M[j]) for j >= i
        # This is the right boundary for subarrays starting at or after i.
        R = [0] * (n + 2)
        R[n + 1] = n + 1
        for i in range(n, 0, -1):
            R[i] = min(R[i + 1], M[i])

        # Base score with all conflicting pairs
        base_score = 0
        for i in range(1, n + 1):
            base_score += R[i] - i
            
        max_score = base_score

        # Consider removing each critical pair [a, M[a]]
        for a in range(1, n):
            if M[a] <= n:
                # Calculate the gain from removing [a, M[a]]
                
                # New R[a] value if M[a] is replaced by M2[a]
                R_new_a = min(R[a + 1], M2[a])
                
                # D[i] = R_new[i] - R[i], the change in the R value at i
                D_a = R_new_a - R[a]
                
                current_gain = D_a
                
                # Propagate the change D backwards from a-1 down to 1
                D_prev = D_a
                for i in range(a, 1, -1):
                    # We are at index i, calculating D for i-1
                    # R[i-1] = min(R[i], M[i-1])
                    if M[i - 1] >= R[i]:  # This implies R[i-1] == R[i]
                        D_curr = min(D_prev, M[i - 1] - R[i])
                    else:  # This implies R[i-1] < R[i], so R[i-1] = M[i-1]
                        # The new R_new[i-1] = min(R_new[i], M[i-1])
                        # Since M[i-1] < R[i] <= R_new[i], R_new[i-1] is also M[i-1].
                        # So, there's no change.
                        D_curr = 0
                    
                    current_gain += D_curr
                    D_prev = D_curr

                max_score = max(max_score, base_score + current_gain)
        
        # If there is only one conflicting pair, removing it makes all subarrays valid.
        if len(conflictingPairs) == 1:
            max_score = max(max_score, n * (n + 1) // 2)

        return max_score