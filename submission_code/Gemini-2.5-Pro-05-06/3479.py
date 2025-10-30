import bisect

class Solution:
  def numberOfSubstrings(self, s: str) -> int:
    N = len(s)
    ans = 0
    
    # zero_indices will store the s-indexes of all '0's.
    # It's augmented with -1 at the beginning and N at the end as sentinels.
    # -1 helps with substrings starting before the first '0'.
    # N helps with substrings ending after the last '0'.
    zero_indices = [-1] 
    for k, char in enumerate(s):
        if char == '0':
            zero_indices.append(k)
    zero_indices.append(N)
    
    # MAX_Z is the maximum number of zeros a substring can have while satisfying
    # ones >= zeros^2. If zeros > sqrt(N), then zeros^2 > N. Since ones <= N,
    # it must be that ones < zeros^2. So, we only need to consider substrings
    # with number of zeros (z) up to floor(sqrt(N)).
    MAX_Z = int(N**0.5)

    # Iterate through all possible starting positions `i` for substrings.
    for i in range(N):
        # Part 1: Count substrings s[i..j] that have ZEROS = 0.
        # These are substrings consisting only of '1's.
        # For such substrings, condition is `ones >= 0*0`, which means `ones >= 0`.
        # This is always true for non-empty substrings of '1's.
        
        # `ptr_to_first_zero_ge_i` is an index into `zero_indices`.
        # `zero_indices[ptr_to_first_zero_ge_i]` gives the s-index of the first '0'
        # at or after `s[i]`. If no such '0', this s-index will be N (the sentinel).
        ptr_to_first_zero_ge_i = bisect.bisect_left(zero_indices, i)
        first_s_idx_zero_ge_i = zero_indices[ptr_to_first_zero_ge_i]
        
        # Substrings s[i...k] where `k` goes from `i` to `first_s_idx_zero_ge_i - 1`
        # are all '1's. The number of such substrings is `first_s_idx_zero_ge_i - i`.
        # E.g., if i=0, first_s_idx_zero_ge_i=3 (s="1110..."), substrings are s[0..0],s[0..1],s[0..2]. Count = 3-0=3.
        # If s[i] is '0', then first_s_idx_zero_ge_i = i, so count is i-i=0. Correct.
        ans += (first_s_idx_zero_ge_i - i)

        # Part 2: Count substrings s[i..j] with ZEROS > 0.
        # Let `z` be the number of zeros in s[i..j]. Iterate `z` from 1 to `MAX_Z`.
        
        # The first '0' at or after s[i] is `zero_indices[ptr_to_first_zero_ge_i]`.
        # The second '0' at or after s[i] is `zero_indices[ptr_to_first_zero_ge_i + 1]`.
        # ...
        # The z-th '0' (1-indexed count) at or after s[i] is `zero_indices[ptr_to_first_zero_ge_i + z - 1]`.
        
        for z in range(1, MAX_Z + 1):
            # `idx_in_zero_indices_for_zth_zero` is the index in `zero_indices` list
            # that holds the s-index of the z-th '0' (for substrings starting at `i`).
            idx_in_zero_indices_for_zth_zero = ptr_to_first_zero_ge_i + z - 1
            
            # Check if this pointer is valid (i.e., points to an actual '0', not the sentinel N or beyond).
            # `len(zero_indices) - 1` is the index of the N sentinel in `zero_indices`.
            # If `idx_in_zero_indices_for_zth_zero` is this index or larger, we've run out of actual '0's.
            if idx_in_zero_indices_for_zth_zero >= len(zero_indices) - 1:
                break 
            
            # `s_idx_of_zth_zero` is the s-index of this z-th '0'.
            s_idx_of_zth_zero = zero_indices[idx_in_zero_indices_for_zth_zero]
            # Note: `s_idx_of_zth_zero == N` check is implicitly handled by the `if` condition above.

            # Consider the substring s[i ... s_idx_of_zth_zero].
            # It starts at `i`, ends at `s_idx_of_zth_zero`, and contains exactly `z` zeros.
            # Length of this substring = `s_idx_of_zth_zero - i + 1`.
            # Number of ones in it = Length - `z`.
            ones_count_up_to_zth_zero = (s_idx_of_zth_zero - i + 1) - z
            
            required_ones = z * z
            
            # To find how many ways this substring can be extended to the right with more '1's
            # while keeping `z` zeros, we find the s-index of the (z+1)-th '0'.
            s_idx_of_next_zero = zero_indices[idx_in_zero_indices_for_zth_zero + 1]
            # `idx_in_zero_indices_for_zth_zero + 1` is a valid index because `idx_in_zero_indices_for_zth_zero`
            # is at most `len(zero_indices) - 2` (index of last actual zero), due to the check above.
            # So `idx_in_zero_indices_for_zth_zero + 1` is at most `len(zero_indices) - 1` (index of sentinel N).
            
            # `max_ones_extendable` is the number of '1's between `s_idx_of_zth_zero` and `s_idx_of_next_zero`.
            # These are '1's at s-indexes from `s_idx_of_zth_zero + 1` to `s_idx_of_next_zero - 1`.
            # Count = (`s_idx_of_next_zero - 1`) - (`s_idx_of_zth_zero + 1`) + 1 
            #       = `s_idx_of_next_zero - s_idx_of_zth_zero - 1`.
            max_ones_extendable = s_idx_of_next_zero - s_idx_of_zth_zero - 1
            
            if ones_count_up_to_zth_zero >= required_ones:
                # The base substring s[i ... s_idx_of_zth_zero] is already valid.
                # Any extension s[i...s_idx_of_zth_zero+k] where characters from
                # s_idx_of_zth_zero+1 to s_idx_of_zth_zero+k are '1's, is also valid.
                # These '1's are from the `max_ones_extendable` block.
                # We can append 0, 1, ..., `max_ones_extendable` '1's.
                # This results in `max_ones_extendable + 1` valid substrings.
                ans += (max_ones_extendable + 1)
            else:
                # The base substring s[i ... s_idx_of_zth_zero] is NOT valid.
                # We need `needed_more_ones` from the `max_ones_extendable` block.
                needed_more_ones = required_ones - ones_count_up_to_zth_zero
                if needed_more_ones <= max_ones_extendable:
                    # We can satisfy the condition by appending `needed_more_ones` or more '1's,
                    # up to `max_ones_extendable` '1's from the block.
                    # Smallest valid substring ends at `s_idx_of_zth_zero + needed_more_ones`.
                    # Largest valid substring ends at `s_idx_of_zth_zero + max_ones_extendable`.
                    # Number of such substrings = `max_ones_extendable - needed_more_ones + 1`.
                    ans += (max_ones_extendable - needed_more_ones + 1)
    return ans