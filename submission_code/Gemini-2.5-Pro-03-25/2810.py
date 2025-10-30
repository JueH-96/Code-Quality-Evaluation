import math
from typing import List

class Solution:
  """
  This class provides a solution to find the minimum cost to collect chocolates of all types.
  """
  def minCost(self, nums: List[int], x: int) -> int:
    """
    Calculates the minimum cost to collect chocolates of all types.

    The problem involves an array `nums` where `nums[i]` is the cost of the chocolate initially at index `i` (which is of type `i`).
    An operation costs `x` and changes the type of the chocolate at index `i` from type `T` to `(T+1) mod n`.
    This is equivalent to saying that after `k` operations, the chocolate at index `j` has cost `nums[j]` and type `(j+k) mod n`.
    We need to find the minimum total cost to collect one chocolate of each type, possibly performing some operations.

    The strategy iterates through the number of operations `k` from 0 to `n-1`. For each `k`, it calculates the total cost associated with performing `k` operations.
    The total cost for a strategy involving `k` operations consists of two parts:
    1. The cost of performing `k` operations: `k * x`.
    2. The minimum cost to collect one chocolate of each type, given that we can choose the state after any number of operations from 0 to `k`.

    Let C(t, k) be the minimum cost to acquire a chocolate of type `t`, considering the states available after performing `p` operations, where `0 <= p <= k`.
    The cost of acquiring type `t` after exactly `p` operations is `nums[(t-p) mod n]`, because the chocolate of type `t` after `p` operations is the one that started at index `(t-p) mod n`.
    Thus, C(t, k) = min_{0 <= p <= k} nums[(t-p) mod n].
    The total collection cost after considering up to `k` operations is `sum(C(t, k) for t in 0..n-1)`.
    The total cost associated with performing `k` operations is `k * x + sum(C(t, k) for t in 0..n-1)`.
    We need to find the minimum of this total cost over all possible values of `k` from 0 to `n-1`. Checking up to `k=n-1` is sufficient because the type configurations repeat every `n` operations, and performing more operations only increases the operation cost without offering potentially lower collection costs than those available within the first `n` states (0 to n-1 operations).

    The algorithm iteratively computes the minimum costs and the total cost for each `k`.

    Args:
        nums: A list of integers representing the initial costs of chocolates at each index.
              The length of the list is n.
        x: The cost of performing one operation.

    Returns:
        The minimum total cost to collect one chocolate of each type.
    """
    n = len(nums)
    
    # min_costs_for_type[t] stores the minimum cost found so far to collect chocolate of type t.
    # This array tracks C(t, k) and is updated iteratively.
    # Initially for k=0 operations, C(t, 0) = nums[t].
    min_costs_for_type = list(nums) 
    
    # Initialize min_total_cost with the total cost for k=0 operations.
    # For k=0, operation cost is 0, collection cost is sum(C(t, 0)).
    min_total_cost = sum(min_costs_for_type) 

    # costs_at_k_ops[t] represents the cost of chocolate type t after exactly k operations.
    # This cost is nums[(t-k) mod n].
    # Initialize for k=0, where costs_at_k_ops[t] = nums[t].
    # This array will be updated in each iteration to represent costs after k operations.
    costs_at_k_ops = list(nums)

    # Iterate through the number of operations k from 1 to n-1.
    for k in range(1, n):
        # Calculate costs_at_k_ops for the current k.
        # This is equivalent to performing a right cyclic shift on the costs_at_k_ops array from the previous step (k-1).
        # The element nums[(t-k) mod n] corresponds to the element nums[((t-1)-(k-1)) mod n] from the previous step, shifted one position.
        
        # Perform one right cyclic shift on costs_at_k_ops array *in-place*.
        # Store the last element which will wrap around to the beginning.
        last_element = costs_at_k_ops[n-1]
        # Shift elements from index n-2 down to 0 one position to the right.
        for i in range(n - 1, 0, -1):
            costs_at_k_ops[i] = costs_at_k_ops[i-1]
        # Place the saved last element at the beginning (index 0).
        costs_at_k_ops[0] = last_element

        # Update the minimum cost found so far for each type: C(t, k) = min(C(t, k-1), costs_at_k_ops[t]).
        # Also, calculate the sum of these minimum costs for the current k: sum(C(t, k) for t in 0..n-1).
        current_collection_sum = 0
        for t in range(n):
            # Update the minimum cost for type t using the cost available after k operations.
            # min_costs_for_type[t] stores C(t, k-1) before update, and C(t, k) after update.
            min_costs_for_type[t] = min(min_costs_for_type[t], costs_at_k_ops[t])
            # Add the updated minimum cost C(t, k) for type t to the current sum.
            current_collection_sum += min_costs_for_type[t]
        
        # Calculate the total cost associated with the strategy of performing k operations.
        # Total cost = (cost of k operations) + (sum of minimum collection costs for each type, considering states 0..k)
        total_k_cost = k * x + current_collection_sum
        
        # Update the overall minimum total cost found across strategies considered so far (for 0..k operations).
        min_total_cost = min(min_total_cost, total_k_cost)
            
    # Return the minimum cost found across all possible numbers of operations k=0..n-1.
    return min_total_cost