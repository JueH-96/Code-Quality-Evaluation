from typing import List

class Solution:
  """
  Solves the Zero Array problem. Checks if an array `nums` can be reduced to all zeros 
  by applying a series of range decrement operations defined by `queries`.
  """
  def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
    """
    Determines if it is possible to transform nums into a Zero Array after processing all the queries sequentially.

    The problem asks if there exists *any* sequence of subset selections during the queries
    that results in the final array being all zeros.

    Let `count[j]` be the total number of queries `i` such that `l_i <= j <= r_i`. This represents
    the maximum number of times the element `nums[j]` can possibly be decremented across all queries.
    A necessary condition for it to be possible to reach a zero array is that for every index `j`,
    the initial value `nums[j]` must not exceed the total number of times it can be decremented, i.e., `nums[j] <= count[j]`.

    This condition turns out to be sufficient as well. A proof can be constructed by considering
    the state `(remaining_value[j], remaining_queries_covering[j])` for each index `j` and showing
    that if the initial condition holds, we can always make valid choices at each query step `i` 
    to maintain the invariant `remaining_value[j] <= remaining_queries_covering[j]`. This is done
    by prioritizing decrements for indices `j` where `remaining_value[j] == remaining_queries_covering[j]`.
    Since the invariant can be maintained, at the end, when `remaining_queries_covering[j]` becomes 0,
    we must have `remaining_value[j] <= 0`. Since values are non-negative, `remaining_value[j]` must be 0.

    Therefore, the problem reduces to calculating `count[j]` for all `j` and checking if `nums[j] <= count[j]`.

    We can calculate `count[j]` efficiently using the difference array technique (also known as prefix sum or sweep line).

    Args:
        nums: An integer array of length n.
        queries: A 2D array where queries[i] = [l_i, r_i] represents the range
                 for the i-th decrement operation. Constraints guarantee 0 <= l_i <= r_i < n.

    Returns:
        True if it is possible to make `nums` a zero array, False otherwise.
    """
    n = len(nums)
    
    # Step 1: Calculate the total number of queries covering each index using a difference array.
    # `diff[i]` will store the net change in the number of active covering intervals at index `i`.
    # The array needs size n+1 because a query ending at index `r` contributes to counts up to `r`,
    # and its effect ends at `r+1`. The change is recorded at `diff[r+1]`.
    diff = [0] * (n + 1) 

    for l, r in queries:
        # A query `[l, r]` increases the coverage count for indices from `l` to `r`.
        # In the difference array representation:
        # Increment the change at the start index `l`.
        diff[l] += 1
        # Decrement the change just after the end index `r`. `r+1` is the index where the interval `[l, r]` ceases to cover.
        # Constraints ensure `r < n`, so `r+1 <= n`. Accessing `diff[r+1]` is valid for `diff` array of size `n+1`.
        diff[r + 1] -= 1

    # Step 2: Compute the prefix sums of the difference array to get the actual counts, 
    # and simultaneously check the condition `nums[j] <= count[j]`.
    current_count = 0 # This variable accumulates the prefix sum, representing count[j] at index j.
    for j in range(n):
        # Update the count for index `j` by adding the net change at this index.
        current_count += diff[j]
        
        # Check the necessary and sufficient condition.
        # If `nums[j]` (the required number of decrements) is greater than `current_count` 
        # (the maximum possible number of decrements for index `j`), then it's impossible.
        if nums[j] > current_count:
            return False

    # If the loop completes without returning False, it means the condition `nums[j] <= count[j]` 
    # holds for all indices `j`. Therefore, it is possible to reach a zero array.
    return True