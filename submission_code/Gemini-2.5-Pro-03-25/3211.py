import bisect
from typing import List
from collections import defaultdict

class Solution:
    """
    Solves the problem of finding the maximum length of a non-decreasing array 
    that can be formed by repeatedly merging subarrays into their sums.
    This problem is equivalent to partitioning the original array `nums` into k contiguous subarrays
    such that their sums B = [b1, b2, ..., bk] form a non-decreasing sequence (b1 <= b2 <= ... <= bk),
    and we want to maximize k.

    The solution uses dynamic programming with optimization.
    Let P be the prefix sum array of `nums`, where P[i] = sum(nums[0...i-1]). P[0] = 0.
    Define dp[i] = (k, S), where k is the maximum length of such a partition for the prefix nums[0...i-1],
    and S is the minimum possible sum of the last segment (bk) among all partitions of length k ending at index i-1.

    To compute dp[i], we consider all possible end points j < i for the second to last segment.
    Suppose the partition ending at j-1 has state dp[j] = (k_j, S_j).
    The last segment for the partition ending at i-1 would be nums[j...i-1].
    Its sum is current_S = P[i] - P[j].
    For the resulting sequence of sums to be non-decreasing, we must have current_S >= S_j.
    Among all valid j, we want to find one that maximizes the new length k_j + 1.
    If there are multiple j giving the same maximum length, we choose the one that minimizes current_S.
    Minimizing current_S = P[i] - P[j] for fixed P[i] is equivalent to maximizing P[j].

    Let V_j = P[j] + S_j. The condition current_S >= S_j is equivalent to P[i] - P[j] >= S_j, which simplifies to P[i] >= P[j] + S_j = V_j.
    So, for a fixed i, we need to find an index j < i such that V_j <= P[i].
    Among all such valid j, we want to find the one that maximizes k_j. Let this maximum length be K'.
    Among all j with k_j = K' and V_j <= P[i], we want to find the one that maximizes P[j]. Let this maximum value be P_max.
    Then the state for index i is dp[i] = (K' + 1, P[i] - P_max).

    Optimization Strategy:
    We maintain, for each possible length k, a list OptPts[k].
    OptPts[k] stores pairs (V_j, P[j]) corresponding to optimal states dp[j] = (k, S_j) where k_j = k.
    These lists store only non-dominated points: if we sort points by V, their P values must be strictly increasing.
    This property allows efficient querying and updates.

    Query: To find the best j for computing dp[i], we iterate k downwards from the maximum k seen so far. For each k, we perform binary search on OptPts[k] to find the point (V_j, P_j) with the largest V_j <= P[i]. Since P values increase with V, this point also has the largest P_j among valid points for this k. The first k that yields a valid point gives the maximum length K'+1.

    Update: After computing dp[i] = (k_i, S_i), we compute the corresponding state information (V_i, P_i) where V_i = P[i] + S_i and P_i = P[i]. We insert this point into OptPts[k_i], maintaining the sorted order by V and the non-dominated property (strictly increasing P). This involves finding the insertion position, checking for domination by the preceding point, and removing any subsequent points dominated by the new point. This update can be done efficiently using list slicing combined with binary search.
    """
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate prefix sums. Python's arbitrary precision integers handle large sums.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
            
        # OptPts[k] stores list of optimal points (V, P) for partitions of length k.
        # V = P[j] + S_j, P = P[j]. S_j is min last sum for length k ending at j-1.
        # Points are maintained sorted by V, and P values are strictly increasing.
        OptPts = defaultdict(list) 
        # Base case: For empty prefix (i=0), partition length is 0, last sum is 0.
        # V = P[0] + S_0 = 0 + 0 = 0. P = P[0] = 0.
        OptPts[0] = [(0, 0)] 
        
        # dp_k[i]: max length k for partition of nums[0...i-1]
        # dp_s[i]: min last sum S for partition of length dp_k[i] ending at i-1
        dp_k = [0] * (n + 1)
        dp_s = [0] * (n + 1)
        # dp_s[0] = 0 aligns with the base case.
        
        max_k_overall = 0 # Tracks the maximum length k achieved across all prefixes processed so far.

        for i in range(1, n + 1):
            current_p = prefix_sum[i] # P[i]
            
            # Find the best previous state (j, k_j, S_j) to extend.
            # Iterate k downwards from max_k_overall.
            found_optimal_prev = False
            for k in range(max_k_overall, -1, -1):
                # Skip if no states recorded for length k.
                if not OptPts[k]: 
                    continue

                k_pts = OptPts[k] # List of (V, P) for length k.
                
                # Perform binary search to find the best point (Vj, Pj) in k_pts such that Vj <= current_p.
                # The 'best' point maximizes Pj among valid points. Due to the structure (V sorted, P increasing),
                # this corresponds to the point with the largest Vj <= current_p.
                
                # bisect_right finds insertion point idx such that all elements before idx have V <= current_p.
                # Using (current_p, float('inf')) ensures correct handling of V == current_p cases.
                insertion_point = bisect.bisect_right(k_pts, (current_p, float('inf')))
                
                if insertion_point == 0:
                    # No point in k_pts satisfies Vj <= current_p.
                    continue # Try smaller k.
                
                # The best point for this k is at index idx = insertion_point - 1.
                idx = insertion_point - 1
                V_j, P_j = k_pts[idx] # This is the optimal (V, P) derived from state dp[j]
                                
                # Found the state corresponding to the maximum possible length k+1 for dp[i].
                dp_k[i] = k + 1
                dp_s[i] = current_p - P_j # Calculate minimum last sum S_i = P[i] - P_j
                
                found_optimal_prev = True
                # Since we iterate k downwards, the first k found is the largest possible k_j. Break loop.
                break 
            
            # If somehow no candidate was found (should not happen with proper base case)
            # if not found_optimal_prev: handle error or edge case. Base case ensures k=0 always works.

            # Update the overall maximum length encountered so far.
            max_k_overall = max(max_k_overall, dp_k[i])
                
            # Prepare the new state information (Vi, Pi) derived from dp[i].
            new_k = dp_k[i] # Length of partition ending at i-1
            new_s = dp_s[i] # Minimum last sum for this length
            new_v = current_p + new_s # Vi = P[i] + Si
            new_p = current_p         # Pi = P[i]
            
            new_point = (new_v, new_p)

            # Update the OptPts list for length new_k, maintaining properties.
            k_pts = OptPts[new_k]
            
            # Maintain sorted by V, strictly increasing P property.
            # Check if new_point is dominated by existing points or dominates existing points.
            
            # Find insertion position p for new_v using bisect_left.
            # bisect_left finds index p such that all elements before p have V < new_v.
            # (new_v, -float('inf')) helps break ties correctly if multiple points have same V.
            p = bisect.bisect_left(k_pts, (new_v, -float('inf'))) 

            # Check if new_point is dominated by the point at index p-1.
            # If P[p-1] >= new_p, it means the existing point is better or equal (smaller V, larger P).
            if p > 0 and k_pts[p-1][1] >= new_p:
                # Dominated, new_point is redundant. Skip adding it.
                continue 

            # Remove points from index p onwards that are dominated by new_point.
            # A point (Vj, Pj) at index j >= p has Vj >= new_v.
            # It is dominated by new_point if Pj <= new_p.
            q = p
            while q < len(k_pts) and k_pts[q][1] <= new_p:
                  # Point k_pts[q] is dominated. Increment q to consider the next point.
                  q += 1
            
            # Replace the slice k_pts[p:q] with the new_point.
            # This removes all dominated points in the range [p, q) and inserts new_point at index p.
            k_pts[p:q] = [new_point]

        # The final answer is the maximum length achieved for the partition of the entire array nums[0...n-1].
        return dp_k[n]