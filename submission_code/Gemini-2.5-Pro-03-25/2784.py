import sys 
from typing import List

# Setting a reasonable recursion depth limit although it's not strictly necessary for iterative solutions.
# sys.setrecursionlimit(2000) 

class Solution:
  def sumOfPower(self, nums: List[int]) -> int:
    """
    Calculates the sum of power of all non-empty groups (subsets) of heroes.
    The power of a group is defined as max(group)^2 * min(group).
    The sum is returned modulo 10^9 + 7.

    Args:
      nums: A list of integers representing hero strengths.

    Returns:
      The sum of powers modulo 10^9 + 7.
    
    Method Explanation:
    Let the sorted array be a_0, a_1, ..., a_{N-1}. The power of a subset S is max(S)^2 * min(S).
    We want to compute the sum of powers over all non-empty subsets S.
    We can group the subsets based on their maximum element. Let's fix the maximum element to be a_j.
    Then any such subset S must contain a_j, and all other elements must be from {a_0, ..., a_{j-1}}.
    So, S = {a_j} union S', where S' is a subset of {a_0, ..., a_{j-1}}.
    For such a subset S, max(S) = a_j. The power is a_j^2 * min(S).
    
    The total sum can be expressed as:
    Sum = Sum_{j=0}^{N-1} Sum_{S' subset {a_0, ..., a_{j-1}}} a_j^2 * min({a_j} union S')
    Let M_j = Sum_{S' subset {a_0, ..., a_{j-1}}} min({a_j} union S').
    The total sum is Sum_{j=0}^{N-1} a_j^2 * M_j.
    
    We derived a relation: M_j = a_j + K_j, where K_j = Sum_{S' subset {a_0, ..., a_{j-1}}, S' != empty} min(S').
    K_j represents the sum of minimums of all non-empty subsets using only the first j elements {a_0, ..., a_{j-1}}.
    We found the recurrence relation K_j = (2 * K_{j-1} + a_{j-1}) % MOD, with K_0 = 0.
    
    The algorithm proceeds as follows:
    1. Sort the input array `nums`. This takes O(N log N) time.
    2. Initialize `current_K = 0` (representing K_0) and `total_sum_of_powers = 0`.
    3. Iterate through the sorted array `nums` from j = 0 to N-1. This loop takes O(N) time.
       a. Get the current element `a_j = nums[j]`.
       b. Calculate `M_j = (a_j + current_K) % MOD`. Here `current_K` holds the value of K_j.
       c. Calculate the contribution of subsets with maximum `a_j`: `term = (a_j^2 * M_j) % MOD`. Compute `a_j^2 % MOD` first to potentially avoid large intermediate numbers.
       d. Add this `term` to `total_sum_of_powers`. Remember to take modulo.
       e. Update `current_K` for the next iteration using the recurrence: `current_K = (2 * current_K + a_j) % MOD`. This computes K_{j+1}.
    4. Return `total_sum_of_powers`.

    The overall time complexity is dominated by sorting, O(N log N). The space complexity is O(N) or O(1) depending on whether sorting is done in-place and its space usage. Typically O(N) or O(log N) for sorting. The iterative calculation uses O(1) extra space.
    """
    
    MOD = 10**9 + 7
    
    # Sort the input array. O(N log N) time complexity.
    nums.sort()
    
    N = len(nums)
    
    # `current_K` stores the sum of minimums of non-empty subsets of elements processed so far.
    # It corresponds to K_j at the start of iteration j (0-indexed). Initialized to K_0 = 0.
    current_K = 0 
    
    # `total_sum_of_powers` accumulates the final result modulo MOD.
    total_sum_of_powers = 0
    
    # Iterate through the sorted array. O(N) time complexity for the loop.
    for j in range(N):
        # Current element being processed.
        a_j = nums[j]
        
        # Calculate M_j = (a_j + K_j) % MOD. `current_K` holds K_j.
        M_j = (a_j + current_K) % MOD
        
        # Calculate the contribution term: a_j^2 * M_j. Perform calculations modulo MOD.
        # Use long long in C++/Java if intermediate products might exceed 32/64 bits. 
        # Python handles large integers automatically. Applying modulo is still important.
        a_j_squared = (a_j * a_j) % MOD # Calculate a_j^2 mod MOD
        term = (a_j_squared * M_j) % MOD # Calculate (a_j^2 * M_j) mod MOD
        
        # Add the contribution to the total sum, ensuring it stays within modulo range.
        total_sum_of_powers = (total_sum_of_powers + term) % MOD
        
        # Update `current_K` for the next iteration (j+1) using the recurrence relation.
        # K_{j+1} = (2 * K_j + a_j) % MOD. `current_K` holds K_j.
        current_K = (2 * current_K + a_j) % MOD
            
    # The final accumulated sum is the answer.
    return total_sum_of_powers