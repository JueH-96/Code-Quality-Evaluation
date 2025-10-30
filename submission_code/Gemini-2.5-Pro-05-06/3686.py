from typing import List

class Solution:
  def beautifulSplits(self, nums: List[int]) -> int:
    n = len(nums)
    if n < 3:
        # Cannot split into three non-empty subarrays
        return 0

    # Hashing parameters
    # nums[i] is 0-50. We use nums[i]+1, so values are 1-51.
    # P1, P2 should be primes greater than 51.
    P1, M1 = 53, 10**9 + 7
    P2, M2 = 61, 10**9 + 9

    # Precompute powers of P1 mod M1 and P2 mod M2
    p_pow1 = [1] * (n + 1)
    p_pow2 = [1] * (n + 1)
    for k in range(1, n + 1):
        p_pow1[k] = (p_pow1[k-1] * P1) % M1
        p_pow2[k] = (p_pow2[k-1] * P2) % M2

    # Precompute prefix hashes
    # hash_valX[k] stores the hash of nums[0...k-1]
    hash_val1 = [0] * (n + 1)
    hash_val2 = [0] * (n + 1)
    for k in range(n):
        val = nums[k] + 1 # Map values to [1, 51] to avoid issues with 0
        hash_val1[k+1] = (hash_val1[k] * P1 + val) % M1
        hash_val2[k+1] = (hash_val2[k] * P2 + val) % M2

    # Helper function to get hash pair for subarray nums[start_idx...end_idx]
    def get_hashes(start_idx: int, end_idx: int): # inclusive indices
        length = end_idx - start_idx + 1
        
        # Calculate hash using (P1, M1)
        h1 = (hash_val1[end_idx+1] - (hash_val1[start_idx] * p_pow1[length]) % M1 + M1) % M1
        # Calculate hash using (P2, M2)
        h2 = (hash_val2[end_idx+1] - (hash_val2[start_idx] * p_pow2[length]) % M2 + M2) % M2
        return (h1, h2)

    count = 0
    # i_cut: first index of nums2. nums1 = nums[0...i_cut-1]
    for i_cut in range(1, n - 1): # i_cut from 1 to n-2
        l1 = i_cut # Length of nums1
        
        # Hash of nums1
        hashes_nums1 = get_hashes(0, l1-1)
        
        # Pre-check for condition 1: nums1 is a prefix of nums[i_cut...]
        # This means H(nums[0...l1-1]) == H(nums[i_cut...i_cut+l1-1])
        # The segment nums[i_cut...i_cut+l1-1] (length l1) must be valid.
        # Its end index is i_cut+l1-1 = 2*i_cut-1. This must be < n, so 2*i_cut <= n.
        is_nums1_prefix_of_shifted_segment = False
        if 2*i_cut <= n: 
            # Hashes of nums[i_cut ... 2*i_cut-1] (which is nums[i_cut ... i_cut+l1-1])
            hashes_target_for_cond1 = get_hashes(i_cut, 2*i_cut-1)
            if hashes_nums1 == hashes_target_for_cond1:
                is_nums1_prefix_of_shifted_segment = True

        # j_cut: first index of nums3. nums2 = nums[i_cut...j_cut-1]
        for j_cut in range(i_cut + 1, n): # j_cut from i_cut+1 to n-1
            l2 = j_cut - i_cut # Length of nums2
            l3 = n - j_cut     # Length of nums3
            
            # All lengths l1, l2, l3 are guaranteed to be >= 1 by loop ranges.

            cond1_met = False
            # Check Condition 1: nums1 is a prefix of nums2
            # Requires len(nums1) <= len(nums2) => l1 <= l2.
            # If true, comparison H(nums1) == H(nums2's prefix of length l1) is needed.
            # This was pre-calculated as is_nums1_prefix_of_shifted_segment.
            # The condition l1 <= l2 (i.e. 2*i_cut <= j_cut) ensures that
            # nums[i_cut...i_cut+l1-1] is part of nums2.
            # Also, if l1 <= l2, then 2*i_cut <= j_cut. Since j_cut < n, 2*i_cut <= n-1 (if j_cut=n-1) or 2*i_cut < n-1.
            # So, the 2*i_cut <= n check for is_nums1_prefix_of_shifted_segment covers this.
            if l1 <= l2 and is_nums1_prefix_of_shifted_segment:
                cond1_met = True
            
            cond2_met = False
            # Check Condition 2: nums2 is a prefix of nums3
            # Requires len(nums2) <= len(nums3) => l2 <= l3.
            # If true, compare H(nums2) with H(nums3's prefix of length l2).
            # The segment nums[j_cut...j_cut+l2-1] (nums3's prefix) must be valid.
            # End index j_cut+l2-1 must be < n, so j_cut+l2 <= n.
            # This is guaranteed by l2 <= n-j_cut, which is exactly l2 <= l3.
            if l2 <= l3:
                hashes_nums2 = get_hashes(i_cut, j_cut - 1)
                hashes_nums3_prefix = get_hashes(j_cut, j_cut + l2 - 1)
                if hashes_nums2 == hashes_nums3_prefix:
                    cond2_met = True
            
            if cond1_met or cond2_met:
                count += 1
    
    return count