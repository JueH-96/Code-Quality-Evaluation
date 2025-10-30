import math
import collections
from typing import List

class Solution:
  """
  Solves the minimum coins problem using dynamic programming with monotonic deque optimization.
  
  Problem Interpretation:
  We are given costs of fruits 1 to N in a 0-indexed array `prices`. `prices[i]` is the cost of fruit `i+1`.
  If we purchase fruit `i` (1-based index), we pay `prices[i-1]` coins and get fruits `i+1, ..., min(N, i+i)` for free.
  We need to find the minimum total cost to acquire all fruits from 1 to N.

  Dynamic Programming Approach:
  Let `dp[i]` be the minimum cost required to acquire fruits `i, i+1, ..., N`. Our goal is to find `dp[1]`.
  The base case is `dp[N+1] = 0`, representing zero cost needed after acquiring fruit N.

  Recurrence Relation Derivation:
  To compute `dp[i]`, we consider the cost if we purchase fruit `i`. The cost incurred is `prices[i-1]`.
  This purchase provides fruits `i+1, ..., min(N, i+i)` for free.
  After purchasing fruit `i`, we must still ensure all fruits from `i+1` onwards are acquired.
  Since fruits `i+1` to `min(N, i+i)` are free, we can choose to take them for free. 
  The next fruit we MUST acquire (either by purchase or ensuring it's covered by a future free offer) could potentially be any fruit `k` starting from `i+1`.
  If we decide to start the next mandatory acquisition step from fruit `k`, the cost from that point onwards is `dp[k]`.
  We can choose the optimal starting point `k` from the range `[i+1, min(N, i+i)+1]`. The index `min(N, i+i)+1` represents the state where we take all offered free fruits and then proceed from the next fruit.
  Therefore, the minimum cost after purchasing fruit `i` is `min(dp[k])` for `k` in the range `[i+1, min(N, i+i)+1]`.
  The total cost `dp[i]` is the cost of purchasing fruit `i` plus this minimum future cost:
  `dp[i] = prices[i-1] + min_{k=i+1}^{\min(N, i+i)+1} dp[k]`

  Optimization with Monotonic Deque:
  The computation requires finding the minimum value in a sliding window `[i+1, R]` where `R = min(N, i+i)+1`.
  As we compute `dp[i]` iterating `i` from `N` down to 1, the window slides and changes size.
  This minimum query can be optimized using a monotonic deque (min-deque).
  The deque stores pairs `(dp_value, index k)` such that `dp_value` is non-decreasing and `index k` is increasing.

  Implementation Details:
  - We use a 1-based index for `dp` array for easier mapping from the problem state `i`. The `dp` array size is `N+2` to accommodate indices `1` to `N+1`.
  - The input `prices` array is 0-indexed, so `prices[i-1]` corresponds to the cost of fruit `i`.
  - The deque is initialized with the base case `(dp[N+1], N+1) = (0, N+1)`.
  - In each step `i`, we adjust the deque window by removing elements outside the range `[i+1, R]` from the left.
  - The minimum value `min_future_cost` is retrieved from the front of the deque.
  - `dp[i]` is computed.
  - The new pair `(dp[i], i)` is added to the deque, maintaining the monotonic property by removing elements from the right.

  Time Complexity: O(N) because each index `i` (from N down to 1) involves constant time operations on the deque (amortized). Each element is added and removed from the deque at most once.
  Space Complexity: O(N) for the `dp` array and the deque.
  """
  def minimumCoins(self, prices: List[int]) -> int:
    N = len(prices)
    
    # dp[i] stores the minimum cost to acquire fruits i, i+1, ..., N.
    # Using 1-based indexing for fruits and dp array helps align with problem statement logic.
    # dp array size N+2 accommodates indices 1 to N+1. Index 0 is unused.
    # dp[N+1] represents the base case: cost to acquire fruits after N, which is 0.
    dp = [float('inf')] * (N + 2)
    dp[N + 1] = 0

    # Monotonic deque storing pairs (dp_value, index k).
    # Stores candidates for the minimum dp[k] in the relevant range efficiently.
    # In the deque, dp_values are non-decreasing, and indices k are increasing.
    dq = collections.deque()
    
    # Initialize deque with the base case: dp[N+1] = 0 at index N+1.
    # This represents the state after acquiring all N fruits.
    dq.append((0, N + 1))

    # Iterate backwards from fruit N down to fruit 1.
    for i in range(N, 0, -1):
        
        # Price of fruit i (1-based index) is prices[i-1] (0-based index in the input list).
        price_i = prices[i-1]
        
        # Determine the right boundary R for the range of k indices [i+1, R].
        # This range corresponds to the possible next states after purchasing fruit i.
        # The fruit indices covered for free are i+1...min(N, i+i).
        # The next state index k can range from i+1 up to min(N, i+i)+1.
        # Let R be the upper bound index for k. Note R could be N+1.
        # The formula correctly handles when i+i >= N because min(N, i+i) = N, so R = N+1.
        R = min(N, i + i) + 1 
        
        # Maintain the deque's window:
        # Remove elements from the front (left) of the deque whose index k is
        # greater than the right boundary R. These states are no longer relevant
        # for the current range minimum query.
        while dq and dq[0][1] > R:
            dq.popleft()

        # After adjusting the window, the minimum value needed (min_future_cost)
        # for the range [i+1, R] is at the front of the deque.
        # Since dq is guaranteed to be non-empty (it starts with (0, N+1) and elements are added),
        # dq[0] is safe to access.
        min_future_cost = dq[0][0]
        
        # Calculate dp[i] using the recurrence relation.
        current_dp_val = price_i + min_future_cost
        dp[i] = current_dp_val
        
        # Add the newly computed dp value for state i to the deque.
        # Before adding (current_dp_val, i), maintain the monotonic property:
        # Remove elements from the back (right) of the deque whose dp_value is
        # greater than or equal to current_dp_val. This ensures that the deque
        # only stores relevant candidates for future minimum queries and keeps values sorted non-decreasingly.
        while dq and dq[-1][0] >= current_dp_val:
             dq.pop()
        # Append the current state's info. The index `i` is associated with `current_dp_val`.
        dq.append((current_dp_val, i))

    # The final answer is dp[1], the minimum cost starting from fruit 1 to acquire all fruits up to N.
    return dp[1]