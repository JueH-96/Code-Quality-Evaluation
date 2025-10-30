class Solution:
  def medianOfUniquenessArray(self, nums: list[int]) -> int:
    n = len(nums)

    # This function counts the number of subarrays with at most k_distinct_limit distinct elements.
    def count_at_most_k_distinct(k_distinct_limit: int) -> int:
      # Max value in nums is 10^5, so max index used in freqs is 100000.
      # Array size 100001 is for indices 0 to 100000.
      # nums[i] are 1-indexed in problem statement, values 1 to 10^5.
      freqs = [0] * 100001 
      
      count_subarrays = 0
      left_ptr = 0
      current_distinct_count = 0
      
      for right_ptr in range(n):
        # Add nums[right_ptr] to the window
        val_right = nums[right_ptr]
        if freqs[val_right] == 0:
          current_distinct_count += 1
        freqs[val_right] += 1
        
        # Shrink window from left if current_distinct_count > k_distinct_limit
        while current_distinct_count > k_distinct_limit:
          val_left = nums[left_ptr]
          freqs[val_left] -= 1
          if freqs[val_left] == 0:
            current_distinct_count -= 1
          left_ptr += 1
        
        # All subarrays ending at 'right_ptr' with a left endpoint 'p' 
        # where left_ptr <= p <= right_ptr have at most k_distinct_limit distinct elements.
        # The number of such valid starting points 'p' is (right_ptr - left_ptr + 1).
        count_subarrays += (right_ptr - left_ptr + 1)
      return count_subarrays

    total_subarrays = n * (n + 1) // 2
    
    # The median is the element at index (L-1)//2 in a 0-indexed sorted uniqueness array.
    # Let this index be m_idx = (L-1)//2.
    # We are looking for the (m_idx + 1)-th smallest value in the uniqueness array.
    # This means we need to find the smallest value 'x' such that
    # the count of subarrays with distinct elements <= x is at least (m_idx + 1).
    median_target_rank = (total_subarrays - 1) // 2 + 1

    # Binary search for the median value.
    # The number of distinct elements in a subarray can range from 1 to n.
    low = 1
    high = n # Max possible distinct elements in any subarray is n.
    ans = n # Initialize with a safe upper bound.

    while low <= high:
      mid_k = low + (high - low) // 2 # Candidate median value
      
      num_subarrays_le_mid_k = count_at_most_k_distinct(mid_k)
      
      if num_subarrays_le_mid_k >= median_target_rank:
        # mid_k could be the median. Or an even smaller value could be.
        # Store mid_k as a potential answer and try to find smaller one.
        ans = mid_k
        high = mid_k - 1 
      else:
        # mid_k is too small. The median value must be larger.
        # The count of subarrays with at most mid_k distinct elements
        # is less than what's needed for mid_k to be the median.
        low = mid_k + 1
        
    return ans