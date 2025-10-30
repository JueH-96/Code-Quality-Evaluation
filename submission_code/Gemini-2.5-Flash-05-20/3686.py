from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        # Choose two large prime moduli and two bases
        # P1, P2 should be distinct primes, ideally greater than max(nums[i]) (50).
        P1 = 53 
        M1 = 1_000_000_007 # A large prime
        P2 = 59 
        M2 = 1_000_000_009 # Another large prime

        # Precompute powers of P for hash calculation
        # p_pow[k] stores P^k mod M
        # p_pow[0] = 1, p_pow[1] = P, p_pow[2] = P^2, ...
        p_pow1 = [1] * (n + 1)
        p_pow2 = [1] * (n + 1)
        for k in range(n):
            p_pow1[k+1] = (p_pow1[k] * P1) % M1
            p_pow2[k+1] = (p_pow2[k] * P2) % M2
        
        # Precompute prefix hashes
        # prefix_hash[k] stores hash of nums[0:k] (i.e., nums[0]...nums[k-1])
        # using the polynomial hashing: S[0]*P^(k-1) + S[1]*P^(k-2) + ... + S[k-1]*P^0
        # It's calculated as prefix_hash[k] = (prefix_hash[k-1] * P + nums[k-1]) % M
        # prefix_hash[0] is 0 (hash of an empty prefix)
        prefix_hash1 = [0] * (n + 1)
        prefix_hash2 = [0] * (n + 1)
        for k in range(n):
            prefix_hash1[k+1] = (prefix_hash1[k] * P1 + nums[k]) % M1
            prefix_hash2[k+1] = (prefix_hash2[k] * P2 + nums[k]) % M2

        # Helper function to get hash of subarray nums[start_idx : end_idx] (exclusive end_idx)
        # The length of the subarray is 'length'.
        # This formula calculates H(S[start_idx...end_idx-1]) = H(S[0...end_idx-1]) - H(S[0...start_idx-1]) * P^length
        def get_sub_hash(start_idx: int, length: int, ph_arr: List[int], pp_arr: List[int], M: int) -> int:
            end_idx = start_idx + length
            # Ensure positive result for modulo operation in Python
            h = (ph_arr[end_idx] - (ph_arr[start_idx] * pp_arr[length]) % M + M) % M
            return h

        count = 0

        # Iterate through all possible split points
        # i represents the length of nums1. nums1 = nums[0:i]
        # Valid range for i: 1 to n-2.
        #   - nums1 must be non-empty (i >= 1)
        #   - nums2 and nums3 combined must have at least 2 elements (n-i >= 2) => i <= n-2
        for i in range(1, n - 1): 

            # j represents the end index (exclusive) of nums2. nums2 = nums[i:j]
            # Valid range for j: i+1 to n-1.
            #   - nums2 must be non-empty (j-i >= 1) => j >= i+1
            #   - nums3 must be non-empty (n-j >= 1) => j <= n-1
            for j in range(i + 1, n): 

                len1 = i
                len2 = j - i
                len3 = n - j

                condition1_met = False
                # Check if nums1 is a prefix of nums2
                # This requires len(nums1) <= len(nums2)
                # And nums[0:len1] must be equal to nums[i : i+len1]
                if len1 <= len2:
                    hash_nums1_1 = get_sub_hash(0, len1, prefix_hash1, p_pow1, M1)
                    hash_nums1_2 = get_sub_hash(0, len1, prefix_hash2, p_pow2, M2)
                    
                    # Hash of the prefix part of nums2 (i.e., the first len1 elements of nums2)
                    hash_nums2_prefix_1 = get_sub_hash(i, len1, prefix_hash1, p_pow1, M1)
                    hash_nums2_prefix_2 = get_sub_hash(i, len1, prefix_hash2, p_pow2, M2)

                    if hash_nums1_1 == hash_nums2_prefix_1 and \
                       hash_nums1_2 == hash_nums2_prefix_2:
                        condition1_met = True

                condition2_met = False
                # Check if nums2 is a prefix of nums3
                # This requires len(nums2) <= len(nums3)
                # And nums[i:j] must be equal to nums[j : j+len2]
                if len2 <= len3:
                    hash_nums2_1 = get_sub_hash(i, len2, prefix_hash1, p_pow1, M1)
                    hash_nums2_2 = get_sub_hash(i, len2, prefix_hash2, p_pow2, M2)

                    # Hash of the prefix part of nums3 (i.e., the first len2 elements of nums3)
                    hash_nums3_prefix_1 = get_sub_hash(j, len2, prefix_hash1, p_pow1, M1)
                    hash_nums3_prefix_2 = get_sub_hash(j, len2, prefix_hash2, p_pow2, M2)

                    if hash_nums2_1 == hash_nums3_prefix_1 and \
                       hash_nums2_2 == hash_nums3_prefix_2:
                        condition2_met = True
                
                if condition1_met or condition2_met:
                    count += 1
        
        return count