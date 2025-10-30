from typing import List
import collections

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        """
        Counts the number of interesting subarrays.
        A subarray nums[l..r] is interesting if the count of indices i in the range [l, r]
        such that nums[i] % modulo == k, is itself congruent to k modulo modulo.

        Let b[i] = 1 if nums[i] % modulo == k, else 0.
        The count of such indices for the subarray nums[l..r] is sum(b[i] for i from l to r).
        The condition for an interesting subarray is (sum(b[i] for i from l to r)) % modulo == k.

        Let S[i] be the prefix sum of the array b, i.e., S[i] = sum(b[0]...b[i]) for i >= 0.
        Define S[-1] = 0.
        The sum of the subarray b[l..r] can be expressed as S[r] - S[l-1].
        The condition becomes (S[r] - S[l-1]) % modulo == k.

        Using modular arithmetic property (a - b) % m == k is equivalent to a % m == (b + k) % m,
        we get S[r] % modulo == (S[l-1] + k) % modulo.
        Alternatively, (a - b) % m == k is equivalent to (a - k) % m == b % m.
        So, (S[r] - k) % modulo == S[l-1] % modulo.

        We iterate through the array with index `i` representing the end index `r` of a subarray (from 0 to n-1).
        For each `r = i`, we want to find the number of valid start indices `l` (from 0 to i).
        Let `j = l-1`. As `l` goes from 0 to i, `j` goes from -1 to i-1.
        The condition is (S[i] - k) % modulo == S[j] % modulo for `j` in [-1, i-1].

        We use a hash map (dictionary) to store the counts of S[p] % modulo values encountered so far,
        for indices `p` strictly less than the current index `i`.
        The key of the map will be `S[p] % modulo`, and the value will be the count of times this modular value has appeared.

        Initialize the map `counts` with the base case `S[-1] % modulo`. S[-1] = 0, so S[-1] % modulo = 0.
        There is one occurrence of S[-1] % modulo = 0 before we start processing index 0.
        `counts = {0: 1}`

        Initialize `total_interesting_subarrays = 0`.
        Initialize `current_prefix_sum = 0`, which will store S[i] in iteration i.

        Iterate through the array from index i = 0 to n-1:
        1. Calculate `b_i = 1` if `nums[i] % modulo == k`, otherwise `b_i = 0`.
        2. Update the current prefix sum: `current_prefix_sum += b_i`. Now `current_prefix_sum` holds `S[i]`.
        3. We are looking for previous indices `j` (where `j = l-1` and `l <= i`) such that `S[j] % modulo == (S[i] - k) % modulo`.
           Let `target_mod_S_j = (current_prefix_sum - k + modulo) % modulo`.
           The number of times `S[j] % modulo` equals `target_mod_S_j` for `j < i` is stored in `counts[target_mod_S_j]`.
           Add this count to `total_interesting_subarrays`.
        4. Update the counts map with the modular prefix sum of the current index `i`:
           `counts[current_prefix_sum % modulo] += 1`. This makes S[i] % modulo available for future iterations (when processing indices > i).

        Finally, return `total_interesting_subarrays`.
        """
        n = len(nums)
        
        # Map to store counts of prefix sum modulo values encountered so far.
        # The keys are `prefix_sum % modulo` values.
        # The values are the number of times that modular value has appeared as S[p] % modulo for p < current_index.
        counts = collections.defaultdict(int)
        
        # Initialize the map with the prefix sum for the imaginary index -1.
        # S[-1] = 0, so S[-1] % modulo = 0.
        # This handles subarrays starting from index 0 (l=0, which corresponds to j = l-1 = -1).
        counts[0] = 1 

        total_interesting_subarrays = 0
        current_prefix_sum = 0 # This variable will store S[i] in iteration i

        # Iterate through the array using index i as the end index (r) of the subarray.
        for i in range(n):
            # Determine if the current element satisfies the condition nums[i] % modulo == k
            is_condition_met = 1 if nums[i] % modulo == k else 0
            
            # Update the current prefix sum S[i]
            # current_prefix_sum holds S[i-1] before this line and S[i] after.
            current_prefix_sum += is_condition_met

            # We are looking for starting indices l (0 <= l <= i) such that subarray nums[l..i] is interesting.
            # The count for subarray nums[l..i] is S[i] - S[l-1].
            # The condition is (S[i] - S[l-1]) % modulo == k.
            # This is equivalent to S[l-1] % modulo == (S[i] - k) % modulo.
            # Let j = l-1. We need to count j in [-1, i-1] such that S[j] % modulo == (S[i] - k + modulo) % modulo.
            
            # Calculate the required value for S[l-1] % modulo
            target_mod_S_l_minus_1 = (current_prefix_sum - k + modulo) % modulo

            # The number of times `target_mod_S_l_minus_1` has occurred as S[j] % modulo for j < i
            # is the number of valid starting indices l for the current ending index i.
            # We retrieve this count from the 'counts' map.
            total_interesting_subarrays += counts[target_mod_S_l_minus_1]

            # Update the count for the current prefix sum modulo value S[i] % modulo.
            # This value will be available for calculating interesting subarrays ending at indices > i.
            counts[current_prefix_sum % modulo] += 1

        return total_interesting_subarrays