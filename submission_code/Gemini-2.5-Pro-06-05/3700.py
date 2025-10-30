import collections
from typing import List

class Solution:
  def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
    MOD = 10**9 + 7
    n = len(nums)
    if n < 5:
      return 0
      
    nums.sort()

    def combinations(n_items, k):
      if k < 0 or k > n_items:
        return 0
      if k == 0 or k == n_items:
        return 1
      if k == 1:
        return n_items
      if k == 2:
        return (n_items * (n_items - 1) // 2)
      return 0

    total_ans = 0

    # l_counts/l_pairs for elements to the left of current middle element
    l_counts = collections.Counter()
    l_pairs = 0
    
    # r_counts/r_pairs for elements to the right of current middle element
    # Initially, all elements are on the right
    r_counts = collections.Counter(nums)
    r_pairs = 0
    for val in r_counts:
      r_pairs += combinations(r_counts[val], 2)

    for j in range(n):
      m = nums[j]

      # Update r_counts and r_pairs for nums[j+1:]
      # nums[j] is no longer on the right side.
      r_pairs -= combinations(r_counts[m], 2)
      r_counts[m] -= 1
      r_pairs += combinations(r_counts[m], 2)
      
      # We need 2 elements from the left and 2 from the right.
      if j >= 2 and n - 1 - j >= 2:
        ans_for_j = 0
        
        m_l = l_counts.get(m, 0)
        other_l_total = j - m_l
        
        m_r = r_counts.get(m, 0)
        other_r_total = (n - 1 - j) - m_r

        # Calculate pairs of non-m elements
        l_pairs_non_m = l_pairs - combinations(l_counts.get(m, 0), 2)
        r_pairs_non_m = r_pairs - combinations(r_counts.get(m, 0), 2)

        # Case c_m = 5: {m,m,m,m,m}
        term = (combinations(m_l, 2) * combinations(m_r, 2))
        ans_for_j += term

        # Case c_m = 4: {m,m,m,m,x}
        term = (combinations(m_l, 2) * combinations(m_r, 1) * other_r_total)
        ans_for_j += term
        term = (combinations(m_l, 1) * other_l_total * combinations(m_r, 2))
        ans_for_j += term

        # Case c_m = 3: {m,m,m,x,y}
        term = (combinations(m_l, 2) * combinations(other_r_total, 2))
        ans_for_j += term
        term = (combinations(m_l, 1) * other_l_total * combinations(m_r, 1) * other_r_total)
        ans_for_j += term
        term = (combinations(other_l_total, 2) * combinations(m_r, 2))
        ans_for_j += term

        # Case c_m = 2: {m,m,x,y,z}, x,y,z distinct
        ways_l_term = combinations(m_l, 1) * other_l_total
        ways_r_distinct_2 = combinations(other_r_total, 2) - r_pairs_non_m
        term = ways_l_term * ways_r_distinct_2
        ans_for_j += term
        
        ways_l_distinct_2 = combinations(other_l_total, 2) - l_pairs_non_m
        ways_r_term = combinations(m_r, 1) * other_r_total
        term = ways_l_distinct_2 * ways_r_term
        ans_for_j += term
        
        total_ans = (total_ans + ans_for_j) % MOD

      # Update l_counts and l_pairs to include nums[j] for the next iteration
      l_pairs -= combinations(l_counts.get(m, 0), 2)
      l_counts[m] += 1
      l_pairs += combinations(l_counts.get(m, 0), 2)
      
    return total_ans