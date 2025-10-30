import collections
from typing import List

class Solution:
  """
  Solves the problem of finding the minimum number of queries k to make the array nums a Zero Array.
  Uses Breadth-First Search (BFS) on the state space defined by the array values and the number of queries processed.
  """
  def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
    """
    Finds the minimum k such that the first k queries can transform nums to a Zero Array.

    Args:
      nums: The initial integer array. Its length n satisfies 1 <= n <= 10.
      queries: A 2D array where each element is [l_i, r_i, val_i]. Its length Q satisfies 1 <= Q <= 1000.

    Returns:
      The minimum non-negative integer k, or -1 if it's impossible. k represents the number of queries used, 
      meaning queries with indices 0 to k-1 are processed.
    """
    n = len(nums)
    q = len(queries)
    
    # Convert the initial list `nums` to a tuple for use as a dictionary key / set element.
    initial_tuple = tuple(nums)
    
    # If the initial array is already all zeros, no queries are needed. k=0.
    if all(x == 0 for x in initial_tuple):
        return 0
    
    # Precompute the maximum possible reduction in sum obtainable from the suffix of queries starting at index i.
    # This is used for pruning the BFS search space. If the current sum of array elements exceeds the
    # maximum possible future reduction, that path cannot lead to the zero array.
    max_future_reduction = [0] * (q + 1)
    # Iterate backwards from the second to last query (index q-1) down to the first query (index 0).
    for i in range(q - 1, -1, -1):
         l, r, val = queries[i]
         # The maximum reduction from query `i` occurs if we select all possible indices in its range [l, r].
         # Each selected index `j` reduces `nums[j]` by `val`.
         # An upper bound on the reduction provided by query `i` is `(r - l + 1) * val`. This assumes
         # all elements in the range are large enough, which might not be true, but it's a valid upper bound.
         query_max_reduction = (r - l + 1) * val 
         # The maximum future reduction from index i onwards is the reduction from query i plus the max from i+1 onwards.
         max_future_reduction[i] = max_future_reduction[i+1] + query_max_reduction

    # Initialize the BFS queue. Each element is a tuple: (current_nums_tuple, k).
    # `current_nums_tuple` represents the state of the array.
    # `k` represents the index of the *next* query to be processed. Initially k=0.
    q_bfs = collections.deque([(initial_tuple, 0)]) 
    
    # Keep track of visited states (represented by array tuples) to avoid cycles and redundant computations.
    # Using a set ensures that we only explore each unique array state once at the earliest possible k.
    visited = {initial_tuple} 

    while q_bfs:
        # Dequeue the next state to explore.
        curr_nums_tuple, k = q_bfs.popleft()

        # Calculate the sum of the current array values. This represents the total reduction still needed.
        current_sum = sum(curr_nums_tuple)
        
        # Pruning check: If the required reduction (current_sum) is greater than the maximum possible
        # reduction achievable by all remaining queries (from index k to q-1), then this path cannot possibly
        # reach the Zero Array state. Skip further exploration from this state.
        # Need k < q because max_future_reduction is indexed up to q.
        if k < q and current_sum > max_future_reduction[k]:
             continue 

        # If we have processed all available queries (k has reached q), we cannot apply more queries.
        # Stop exploring this path. Note: the check if curr_nums_tuple is zero happens when states are generated.
        if k == q:
            continue

        # Get details of the k-th query (0-indexed).
        l, r, val = queries[k]
        
        # Identify the list of indices within the query's range [l, r].
        indices_in_range = [j for j in range(l, r + 1)]
        num_indices = len(indices_in_range)

        # Iterate through all possible subsets of indices within the range [l, r].
        # There are 2^num_indices subsets. We use bit manipulation to represent subsets.
        # `i_subset` iterates from 0 to 2^num_indices - 1. Each bit corresponds to an index in `indices_in_range`.
        for i_subset in range(1 << num_indices):
            # Start with a mutable list copy of the current array state tuple.
            temp_next_nums = list(curr_nums_tuple)
            # Flag to track if the current subset choice is valid (respects non-negativity).
            valid_subset = True
            
            # Apply the decrements based on the chosen subset S_k represented by `i_subset`.
            for bit_pos in range(num_indices):
                # Check if the index corresponding to `bit_pos` is included in the current subset.
                if (i_subset >> bit_pos) & 1:
                    # Get the actual array index `idx_j`.
                    idx_j = indices_in_range[bit_pos]
                    
                    # Crucial check: Ensure we don't decrement `nums[j]` below zero.
                    # If `nums[j]` is less than `val`, we cannot apply the decrement for this index.
                    if temp_next_nums[idx_j] < val:
                        # Mark the subset choice as invalid and break the inner loop.
                        valid_subset = False
                        break 
                    
                    # Perform the decrement since it's valid.
                    temp_next_nums[idx_j] -= val 
            
            # If the subset application was valid for all chosen indices
            if valid_subset:
                # Convert the resulting list state back to an immutable tuple for use in the visited set.
                next_nums_tuple = tuple(temp_next_nums)

                # If this resulting state (array tuple) has not been visited before
                if next_nums_tuple not in visited:
                     
                     # Check if we have reached the target Zero Array state (all elements are 0).
                     if all(x == 0 for x in next_nums_tuple):
                       # Successfully reached the zero state after processing query k (index k).
                       # This means k+1 queries (indices 0 to k) were used in total.
                       # Since BFS explores level by level, this is the minimum number of queries.
                       return k + 1

                     # Mark the new state as visited to avoid re-exploring it.
                     visited.add(next_nums_tuple)
                     # Add the new state to the BFS queue. The next query to process from this state will be k+1.
                     q_bfs.append((next_nums_tuple, k + 1))
    
    # If the BFS queue becomes empty and we haven't returned a value, 
    # it means the Zero Array state is unreachable with the given queries.
    return -1