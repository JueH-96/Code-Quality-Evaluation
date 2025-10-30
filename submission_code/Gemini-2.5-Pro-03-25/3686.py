from typing import List
# import random # Not needed as we use fixed bases

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        """
        Counts the number of beautiful splits of the array nums.
        A split is beautiful if nums is split into three non-empty subarrays nums1, nums2, nums3
        such that nums = nums1 + nums2 + nums3 (concatenated), and either nums1 is a prefix of nums2
        or nums2 is a prefix of nums3.

        Args:
            nums: A list of integers.

        Returns:
            The number of beautiful splits.
        
        The solution uses polynomial rolling hash to efficiently check prefix conditions.
        Two hash functions with different bases and large prime moduli are used to minimize
        the probability of hash collisions.
        Time complexity: O(n^2), where n is the length of nums. This is due to iterating through
                         all possible pairs of split points (i, j). Hash computation and comparison
                         take O(1) time after O(n) precomputation.
        Space complexity: O(n) for storing precomputed powers and prefix hashes.
        """
        n = len(nums)
        
        # A split requires three non-empty subarrays. 
        # The minimum length of nums for a valid split is 1 + 1 + 1 = 3.
        if n < 3:
            return 0

        # Define parameters for the two hash functions.
        # Using large prime moduli helps in reducing collisions.
        MOD1 = 10**9 + 7
        MOD2 = 10**9 + 9
        
        # Choose prime bases for hashing. The bases should be larger than the maximum possible 
        # element value after adjustment (max(nums[i]) + 1 = 50 + 1 = 51).
        # Fixed bases are used for simplicity and reproducibility.
        P1 = 53  # A prime number > 51
        P2 = 61  # Another prime number > 51

        # Precompute powers of P1 modulo MOD1 and P2 modulo MOD2.
        # pow1[k] stores P1^k % MOD1
        # pow2[k] stores P2^k % MOD2
        pow1 = [1] * (n + 1)
        pow2 = [1] * (n + 1)
        for k in range(1, n + 1):
            pow1[k] = (pow1[k - 1] * P1) % MOD1
            pow2[k] = (pow2[k - 1] * P2) % MOD2

        # Precompute prefix hashes for the array.
        # We add 1 to each element `nums[k]` to map the values to the range [1, 51].
        # This avoids potential issues if `nums[k]` could be 0 and ensures all elements contribute positively.
        # prefix_hash1[k] stores the hash of the prefix nums[0:k] using (P1, MOD1).
        # prefix_hash2[k] stores the hash of the prefix nums[0:k] using (P2, MOD2).
        prefix_hash1 = [0] * (n + 1)
        prefix_hash2 = [0] * (n + 1)
        for k in range(n):
            val = nums[k] + 1  # Map values to the range [1, 51]
            prefix_hash1[k + 1] = (prefix_hash1[k] * P1 + val) % MOD1
            prefix_hash2[k + 1] = (prefix_hash2[k] * P2 + val) % MOD2

        # Helper function to compute the hash pair (hash1, hash2) for the subarray nums[l:r].
        # The interval is [l, r), meaning index l is inclusive and index r is exclusive.
        def get_hash(l, r):
            # If the range is invalid or empty, return hash (0, 0).
            if l >= r:
                return (0, 0)
            
            # Calculate the length of the subarray.
            length = r - l
            
            # Calculate hash using the first pair (P1, MOD1).
            # The formula hash(l, r) = (prefix_hash[r] - prefix_hash[l] * P^length) % MOD is used.
            h1 = (prefix_hash1[r] - prefix_hash1[l] * pow1[length]) % MOD1
            # Ensure the result is non-negative (Python's % operator can return negative results).
            if h1 < 0: h1 += MOD1
                
            # Calculate hash using the second pair (P2, MOD2).
            h2 = (prefix_hash2[r] - prefix_hash2[l] * pow2[length]) % MOD2
            # Ensure the result is non-negative.
            if h2 < 0: h2 += MOD2
            
            # Return the pair of hashes. Comparing pairs significantly reduces collision probability.
            return (h1, h2)

        count = 0
        # Iterate through all possible pairs of split points (i, j).
        # 'i' is the index marking the end of nums1 (exclusive) and the start of nums2 (inclusive).
        # 'j' is the index marking the end of nums2 (exclusive) and the start of nums3 (inclusive).
        # The loop ranges `1 <= i <= n-2` and `i+1 <= j <= n-1` ensure that `1 <= i < j < n`.
        # This guarantees that nums1, nums2, and nums3 are all non-empty.
        for i in range(1, n - 1):  
            for j in range(i + 1, n):  
                
                # Calculate the lengths of the three potential subarrays based on split points i and j.
                len1 = i      # Length of nums1 = nums[0:i]
                len2 = j - i  # Length of nums2 = nums[i:j]
                len3 = n - j  # Length of nums3 = nums[j:n]

                # Condition 1: Check if nums1 is a prefix of nums2.
                # This check is only relevant if length(nums2) >= length(nums1).
                is_prefix12 = False
                if len2 >= len1:
                    # Compare the hash of nums1 (subarray nums[0:i]) with the hash of the 
                    # first len1 elements of nums2 (subarray nums[i : i + len1] which simplifies to nums[i : 2*i]).
                    if get_hash(0, i) == get_hash(i, 2 * i):
                        is_prefix12 = True

                # Condition 2: Check if nums2 is a prefix of nums3.
                # This check is only relevant if length(nums3) >= length(nums2).
                is_prefix23 = False
                if len3 >= len2:
                    # Compare the hash of nums2 (subarray nums[i:j]) with the hash of the 
                    # first len2 elements of nums3 (subarray nums[j : j + len2] which simplifies to nums[j : 2*j - i]).
                    if get_hash(i, j) == get_hash(j, 2 * j - i):
                        is_prefix23 = True

                # A split defined by (i, j) is beautiful if either of the prefix conditions holds true.
                if is_prefix12 or is_prefix23:
                    count += 1

        return count