from typing import List

class Solution:
  def numberOfGoodPartitions(self, nums: List[int]) -> int:
    MOD = 1_000_000_007
    n = len(nums)

    # Constraints: 1 <= nums.length <= 10^5. So n >= 1.
    # If n == 1, the logic below correctly results in num_segments = 1.
    # Then pow(2, 1 - 1, MOD) = pow(2, 0, MOD) = 1, which is correct.

    # Step 1: Precompute the last occurrence index for each number in nums.
    last_occurrence = {}
    for i, num in enumerate(nums):
        last_occurrence[num] = i

    # Step 2: Identify the minimal, non-overlapping segments (blocks).
    num_segments = 0
    
    # current_idx_in_nums is the starting index of the current segment being identified.
    current_idx_in_nums = 0 
    while current_idx_in_nums < n:
        num_segments += 1 # Found the start of a new segment
        
        # The current segment starts at current_idx_in_nums.
        # It must extend at least to the last occurrence of nums[current_idx_in_nums].
        current_segment_max_reach = last_occurrence[nums[current_idx_in_nums]]
        
        # Optimization: If this initial segment_max_reach already covers the rest of the array,
        # then this segment is the final one. We can stop processing further segments.
        # The loop will terminate naturally after this iteration because
        # current_idx_in_nums will be updated to current_segment_max_reach + 1 = (n-1) + 1 = n.
        # (This break is for the outer `while current_idx_in_nums < n` loop)
        if current_segment_max_reach == n - 1:
            break 

        # Scan elements within the current conceptual segment boundaries:
        # nums[current_idx_in_nums ... current_segment_max_reach].
        # The element nums[current_idx_in_nums] itself has already been considered
        # to initialize current_segment_max_reach. So, start scan_ptr from next element.
        scan_ptr = current_idx_in_nums + 1
        while scan_ptr <= current_segment_max_reach:
            # If an element nums[scan_ptr] must extend beyond current_segment_max_reach,
            # update current_segment_max_reach to this new furthermost index.
            if last_occurrence[nums[scan_ptr]] > current_segment_max_reach:
                current_segment_max_reach = last_occurrence[nums[scan_ptr]]
            
                # Optimization: If current_segment_max_reach is now extended to the end of the array,
                # this segment becomes the final one. No need to scan further within this segment.
                # The outer loop will terminate correctly after this segment is processed.
                # (This break is for the inner `while scan_ptr <= current_segment_max_reach` loop)
                if current_segment_max_reach == n - 1:
                    break 
            
            scan_ptr += 1
        
        # The current segment ends at current_segment_max_reach.
        # The next segment must start at current_segment_max_reach + 1.
        current_idx_in_nums = current_segment_max_reach + 1
            
    # Step 3: Calculate the number of ways to partition these segments.
    # If there are `k` segments, there are `k-1` places to make a cut.
    # Each place can either have a cut or not, so 2^(k-1) ways.
    # `num_segments` will always be at least 1 because n >= 1.
    # So, `num_segments - 1` will be >= 0.
    return pow(2, num_segments - 1, MOD)