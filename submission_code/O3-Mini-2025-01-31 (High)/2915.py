from typing import List
import collections

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # The idea:
        # Define an indicator for each element: indicator[i] = 1 if nums[i] % modulo == k, else 0.
        # Let prefix be the cumulative sum of these indicator values.
        # For a subarray nums[l..r], the count of indices satisfying the condition is
        #   c = prefix[r] - (prefix[l-1] if l > 0 else 0).
        # We need c % modulo == k.
        #
        # Using modulo arithmetic, we have:
        #   (prefix[r] - prefix[l-1]) % modulo == k
        # <=> prefix[r] % modulo - prefix[l-1] % modulo ≡ k (mod modulo)
        #
        # For each index r, if we let r_rem = prefix[r] % modulo, then valid l are those where
        #   prefix[l-1] % modulo == (r_rem - k) % modulo.
        #
        # We use a dictionary (frequency map) to track the counts of prefix remainders (mod modulo)
        # that have been seen so far. Initially, we consider prefix sum 0 (before processing any element).
        
        freq = collections.defaultdict(int)
        freq[0] = 1  # before any element, prefix sum = 0.
        prefix = 0
        ans = 0
        
        for num in nums:
            # Update prefix sum with the indicator (1 if condition holds, else 0)
            if num % modulo == k:
                prefix += 1
            else:
                prefix += 0  # no change
            
            # Compute current prefix remainder modulo modulo.
            current_rem = prefix % modulo
            
            # Find the target remainder needed for the previous prefix so that the subarray formed is interesting.
            target = (current_rem - k) % modulo
            
            # Add the number of previous prefixes that have this remainder.
            ans += freq[target]
            
            # Update the frequency count for the current remainder.
            freq[current_rem] += 1
        
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    # nums = [3,2,4], modulo = 2, k = 1 produces interesting subarrays:
    # [3], [3,2], [3,2,4]  so the answer is 3.
    print(sol.countInterestingSubarrays([3,2,4], 2, 1))  # Expected output: 3

    # Example 2:
    # nums = [3,1,9,6], modulo = 3, k = 0 produces interesting subarrays:
    # [3,1,9,6] and [1] so the answer is 2.
    print(sol.countInterestingSubarrays([3,1,9,6], 3, 0))  # Expected output: 2