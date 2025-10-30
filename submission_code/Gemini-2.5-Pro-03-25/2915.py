import collections
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        """
        Counts the number of interesting subarrays using a prefix sum approach
        combined with a hash map (dictionary).

        An interesting subarray nums[l..r] is defined by the condition:
        cnt % modulo == k, where cnt is the number of elements nums[i] (l <= i <= r)
        such that nums[i] % modulo == k.

        Args:
            nums: The input list of integers.
            modulo: The modulo value for the conditions.
            k: The target remainder value for the conditions.

        Returns:
            The total count of interesting subarrays.
        
        The core idea is based on prefix sums and remainders.
        Let P[i] be the count of elements nums[j] such that nums[j] % modulo == k for 0 <= j < i.
        P[0] = 0.
        The count for a subarray nums[l..r] is C = P[r+1] - P[l].
        We are looking for subarrays where C % modulo == k.
        Substituting C, we get (P[r+1] - P[l]) % modulo == k.
        This can be rewritten as P[l] % modulo == (P[r+1] - k) % modulo.

        We iterate through the array, maintaining the current prefix count P[i+1] as we process nums[i].
        For each index i (representing the right end r of potential subarrays), we calculate P[i+1].
        We then determine the target remainder required for the starting prefix P[l]: 
        target_rem = (P[i+1] - k + modulo) % modulo.
        We use a hash map (dictionary `counts`) to store the frequencies of the remainders of the prefix counts encountered so far (P[0] % modulo, P[1] % modulo, ..., P[i] % modulo).
        The value `counts[target_rem]` tells us how many valid starting indices l (where l <= i) exist such that P[l] % modulo == target_rem. We add this number to our total count of interesting subarrays.
        Finally, we update the `counts` map with the remainder of the current prefix count P[i+1] % modulo.
        """
        n = len(nums)
        total_interesting_subarrays = 0
        
        # counts stores the frequency of (prefix_count % modulo) encountered so far.
        # The key is the remainder (P[j] % modulo), and the value is its frequency.
        # Initialize with {0: 1} to account for the empty prefix P[0] = 0.
        # P[0] % modulo = 0. This initial entry handles subarrays starting at index 0.
        counts = collections.defaultdict(int)
        counts[0] = 1
        
        # current_prefix_count tracks the cumulative count of elements satisfying
        # nums[j] % modulo == k for j from 0 up to the current index i (inclusive).
        # This corresponds to P[i+1] in the prefix sum notation defined above.
        current_prefix_count = 0 
        
        for i in range(n):
            # Update the prefix count based on the current element nums[i].
            # Increment the count if the element satisfies the condition nums[i] % modulo == k.
            if nums[i] % modulo == k:
                current_prefix_count += 1
            
            # Calculate the remainder of the current prefix count modulo `modulo`.
            # This is P[i+1] % modulo.
            current_rem = current_prefix_count % modulo
            
            # Calculate the target remainder for the starting prefix P[l].
            # We need P[l] % modulo == (P[i+1] - k) % modulo.
            # target_rem = (P[i+1] % modulo - k % modulo + modulo) % modulo
            # Since current_rem = P[i+1] % modulo, this simplifies to:
            # target_rem = (current_rem - k + modulo) % modulo
            # The "+ modulo" ensures the result of (current_rem - k) is non-negative before the final modulo operation.
            target_rem = (current_rem - k + modulo) % modulo
            
            # Add the count of prefixes P[l] (for l <= i) that have the target remainder.
            # counts[target_rem] gives the number of valid starting points l (indices corresponding to the start of the prefix P[l])
            # for subarrays ending at index i. Each such l corresponds to an interesting subarray nums[l..i].
            total_interesting_subarrays += counts[target_rem]
            
            # Update the counts dictionary with the remainder of the current prefix P[i+1].
            # This prefix count P[i+1] (with remainder current_rem) corresponds to index i+1 in the prefix array P.
            # It can serve as a starting prefix P[l] for future subarrays ending at indices > i.
            counts[current_rem] += 1
            
        return total_interesting_subarrays