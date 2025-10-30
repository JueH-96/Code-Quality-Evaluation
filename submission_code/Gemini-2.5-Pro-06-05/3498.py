from typing import List

class Solution:
  def minChanges(self, nums: List[int], k: int) -> int:
    """
    Calculates the minimum changes to make the array k-semipalindromic.
    A k-semipalindrome has a constant difference X between symmetric elements.
    """
    n = len(nums)
    num_pairs = n // 2

    # diff_freq[d] stores the number of pairs (u, v)
    # with an absolute difference of d.
    diff_freq = [0] * (k + 1)
    
    # max_bound_freq[m] will store the number of pairs for which
    # the maximum achievable difference with at most one change is m.
    # m = max(u, v, k-u, k-v)
    max_bound_freq = [0] * (k + 1)

    for i in range(num_pairs):
        u, v = nums[i], nums[n - 1 - i]
        
        d = abs(u - v)
        diff_freq[d] += 1
        
        # This is the maximum difference we can get with at most one change.
        m = max(u, v, k - u, k - v)
        max_bound_freq[m] += 1
    
    # Convert max_bound_freq to a suffix sum array.
    # After this, max_bound_freq[x] will represent the number of pairs
    # that can be changed to have a difference of x with at most one change.
    for i in range(k - 1, -1, -1):
        max_bound_freq[i] += max_bound_freq[i+1]
        
    # To minimize Cost(X) = 2*N - (2*C0(X) + C1(X)), we must maximize
    # 2*C0(X) + C1(X).
    # This simplifies to maximizing diff_freq[X] + one_change_possible_count[X].
    # Let's call this `max_benefit`.
    
    max_benefit = 0
    for X in range(k + 1):
        # one_change_possible_count[X] is stored in max_bound_freq[X] after suffix sum
        benefit = diff_freq[X] + max_bound_freq[X]
        if benefit > max_benefit:
            max_benefit = benefit
            
    # The minimum cost is the maximum possible cost (2 changes per pair)
    # minus the maximum benefit we can get for some X.
    # Max possible cost is 2 * num_pairs = n.
    return n - max_benefit