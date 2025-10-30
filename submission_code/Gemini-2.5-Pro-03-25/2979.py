import sys
# Setting a higher recursion depth is generally not needed for iterative solutions 
# like this DP, but uncomment if specific environments require it.
# sys.setrecursionlimit(200000) 

from typing import List

class Solution:
  """
    Solves the problem of maximizing profit from selling houses based on offers using dynamic programming.
    
    The problem is equivalent to finding the maximum weight subset of non-overlapping intervals on a number line,
    where each offer [start, end, gold] corresponds to an interval [start, end] with weight 'gold'.
    
    We use dynamic programming on the houses. The state dp[i] represents the maximum profit 
    that can be achieved considering houses from index 0 up to index i-1.
    
    The time complexity is O(N + M), where N = n is the number of houses and M = len(offers) is the number of offers.
    This comes from O(M) to process offers and O(N + M) for the DP loop (each offer is processed once).
    
    The space complexity is O(N + M), primarily for the DP table `dp` of size N+1 and the `offers_by_end` structure 
    which stores M offer details distributed across N lists.
  """
  def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
    """
    Calculates the maximum gold earned by selecting a set of non-overlapping offers.

    Args:
        n: The number of houses, numbered from 0 to n-1.
        offers: A list of offers, where each offer is represented as [start_i, end_i, gold_i],
                indicating a buyer wants houses from start_i to end_i for gold_i amount.

    Returns:
        The maximum amount of gold that can be earned by selecting a subset of non-overlapping offers.
    """
    
    # Step 1: Preprocess offers to group them by their end index.
    # This allows for efficient retrieval of all offers ending at a particular house index `k`.
    # `offers_by_end[k]` will store a list of pairs `[start, gold]` for all offers ending at index `k`.
    # We use a list of lists for this grouping. The outer list has size n.
    offers_by_end = [[] for _ in range(n)]
    for start, end, gold in offers:
        # Append the [start index, gold amount] pair to the list corresponding to the offer's end index.
        offers_by_end[end].append([start, gold])

    # Step 2: Initialize the Dynamic Programming (DP) array.
    # `dp[i]` will store the maximum profit achievable considering houses from 0 up to `i-1`.
    # The array `dp` has size `n+1` to cover states from 0 houses (index 0) up to `n` houses (index n, representing houses 0..n-1).
    dp = [0] * (n + 1)
    # The base case `dp[0] = 0` (0 profit for 0 houses) is implicitly set by initializing with zeros.
    
    # Step 3: Fill the DP table iteratively.
    # We iterate from `i = 1` to `n`. In each iteration `i`, we calculate `dp[i]`.
    for i in range(1, n + 1):
        # The house index currently being considered as a potential endpoint for an offer is `end_idx = i - 1`.
        end_idx = i - 1 
        
        # Option 1: Don't accept any offer ending exactly at `end_idx`.
        # In this case, the maximum profit is the same as the maximum profit achieved considering houses up to `i-2`.
        # This value is already computed and stored in `dp[i-1]`.
        # We initialize `dp[i]` with this value. It serves as the baseline profit for state `i`.
        dp[i] = dp[i-1]
        
        # Option 2: Consider accepting an offer that ends exactly at `end_idx`.
        # We check if there are any offers registered that end at this `end_idx`.
        # `offers_by_end[end_idx]` holds the list of such offers. If this list is not empty:
        if offers_by_end[end_idx]: 
            # Iterate through all offers `[start, gold]` that end at `end_idx`.
            for start, gold in offers_by_end[end_idx]:
                # If we choose to accept this offer `[start, end_idx, gold]`:
                # The total profit obtained would be the `gold` from this offer, plus
                # the maximum profit achievable using houses *before* the start of this offer, 
                # i.e., houses from 0 up to `start-1`. This maximum profit is stored in `dp[start]`.
                profit_with_this_offer = dp[start] + gold
                
                # We want to maximize the profit for `dp[i]`. We compare the current value of `dp[i]`
                # (which represents the max profit found so far for houses 0..i-1, potentially including other offers ending at end_idx)
                # with the profit obtainable by accepting *this* specific offer. We take the maximum.
                dp[i] = max(dp[i], profit_with_this_offer)
        
    # Step 4: Return the final result.
    # After the loop completes, `dp[n]` contains the maximum profit considering all houses from 0 to n-1.
    return dp[n]