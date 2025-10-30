from typing import List

class Solution:
  def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
    # dp[i] stores the maximum profit considering houses from index 0 to i-1.
    # The actual houses are indexed 0 to n-1.
    # The dp array is of size n+1.
    # dp[0] represents the profit from an empty set of houses, which is 0.
    # dp[k] represents the maximum profit considering houses 0, ..., k-1.
    # The final answer will be dp[n], representing max profit for houses 0, ..., n-1.
    dp = [0] * (n + 1)

    # Group offers by their end_i.
    # offers_grouped_by_end[j] will store a list of tuples (start_k, gold_k)
    # for all offers that end at house j.
    # House indices are 0 to n-1.
    # So, offers_grouped_by_end will be a list of lists, with length n.
    offers_grouped_by_end = [[] for _ in range(n)]
    for start_i, end_i, gold_i in offers:
        # Store (start_index, gold_value) for offers ending at end_i
        offers_grouped_by_end[end_i].append((start_i, gold_i))

    # Iterate from dp_idx = 1 to n.
    # dp_idx corresponds to the state "after" considering house dp_idx-1.
    # Thus, dp[dp_idx] is the max profit for houses 0, ..., dp_idx-1.
    # The current house that might be an endpoint of an offer is house dp_idx-1.
    for dp_idx in range(1, n + 1):
        # Option 1: House (dp_idx-1) is not sold, or it is sold as part of an offer
        # that ended before (dp_idx-1), or simply this house is not the *endpoint*
        # of any offer we choose at this step.
        # In this scenario, the profit is the same as the maximum profit obtained
        # considering houses 0, ..., dp_idx-2. This value is stored in dp[dp_idx-1].
        dp[dp_idx] = dp[dp_idx - 1]

        # Option 2: House (dp_idx-1) is sold as the endpoint of one or more offers.
        # current_house_idx_as_endpoint is the actual index (0 to n-1) of this house.
        current_house_idx_as_endpoint = dp_idx - 1
        
        # Iterate through all offers that end at current_house_idx_as_endpoint.
        # These offers are of the form [start_val, current_house_idx_as_endpoint, gold_val].
        # The check `if offers_grouped_by_end[current_house_idx_as_endpoint]:` is not strictly
        # necessary as iterating over an empty list does nothing, but it's harmless.
        for start_val, gold_val in offers_grouped_by_end[current_house_idx_as_endpoint]:
            # If we choose to accept this offer:
            # The gold obtained from this specific offer is gold_val.
            # This offer covers houses from start_val to current_house_idx_as_endpoint.
            # The maximum profit from houses 0, ..., start_val-1 (i.e., houses before
            # this offer begins) must have been accumulated already. This profit is
            # stored in dp[start_val]. (Recall dp[k] is for houses 0..k-1).
            profit_if_take_this_offer = dp[start_val] + gold_val
            
            # We want to maximize dp[dp_idx].
            dp[dp_idx] = max(dp[dp_idx], profit_if_take_this_offer)
    
    return dp[n]