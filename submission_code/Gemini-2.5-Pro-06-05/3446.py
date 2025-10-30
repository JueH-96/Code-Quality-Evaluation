class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        import collections

        # The condition for a good pair (i, j) is nums1[i] % (nums2[j] * k) == 0.
        # This implies that nums1[i] must be a multiple of k.
        # If so, let target = nums1[i] // k. The condition simplifies to target % nums2[j] == 0,
        # which means nums2[j] must be a divisor of target.
        #
        # Due to the small constraints on values (1 to 50), we can use precomputation.
        # We create an array `divisors_count` where `divisors_count[t]` will store
        # the number of elements in nums2 that are divisors of `t`.

        MAX_VAL = 50
        divisors_count = [0] * (MAX_VAL + 1)

        # Count frequencies of numbers in nums2 to handle duplicates efficiently.
        freq_nums2 = collections.Counter(nums2)

        # Populate `divisors_count`. For each number `val` from nums2,
        # it is a divisor of all its multiples. We add its frequency
        # to the count for each of its multiples up to MAX_VAL.
        for val, freq in freq_nums2.items():
            for multiple in range(val, MAX_VAL + 1, val):
                divisors_count[multiple] += freq

        # Calculate the total number of good pairs by iterating through nums1.
        total_pairs = 0
        for num1 in nums1:
            if num1 % k == 0:
                target = num1 // k
                # The number of good pairs for this num1 is the number of elements
                # in nums2 that divide `target`. This is precomputed.
                # Since 1 <= num1, k <= 50, if num1 is a multiple of k, then target >= 1.
                # Also, target <= 50, so it's a valid index.
                total_pairs += divisors_count[target]

        return total_pairs