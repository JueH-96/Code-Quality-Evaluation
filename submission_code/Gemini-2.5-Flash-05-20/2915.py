import collections
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        """
        Counts the number of "interesting" subarrays in nums.

        A subarray nums[l..r] is interesting if the count of elements x
        in nums[l..r] such that x % modulo == k, satisfies (count % modulo == k).

        The solution transforms the problem into a prefix sum problem on a binary array.
        Let A[i] = 1 if nums[i] % modulo == k, else A[i] = 0.
        We need to find subarrays A[l..r] whose sum (cnt) satisfies cnt % modulo == k.

        This is solved using a hash map to store frequencies of prefix sum remainders.
        Let P[i] be the prefix sum of A up to index i-1 (P[0] = 0).
        The sum of A[l..r] is P[r+1] - P[l].
        We need (P[r+1] - P[l]) % modulo == k.
        This is equivalent to: (P[r+1] % modulo - P[l] % modulo + modulo) % modulo == k.
        So, P[l] % modulo must be (P[r+1] % modulo - k + modulo) % modulo.
        """
        
        total_interesting_subarrays = 0
        
        # rem_counts stores the frequency of (prefix_sum % modulo) values encountered so far.
        # Initialize with {0: 1} to account for the empty prefix sum (0) at index -1.
        # This handles subarrays that start from index 0.
        rem_counts = collections.defaultdict(int)
        rem_counts[0] = 1 
        
        # current_prefix_sum_val keeps track of P[i+1] (sum of A[0]...A[i])
        current_prefix_sum_val = 0
        
        for i in range(len(nums)):
            # Step 1: Determine A_i based on nums[i]
            A_i = 1 if nums[i] % modulo == k else 0
            
            # Step 2: Update current_prefix_sum_val (sum of A[0]...A[i])
            current_prefix_sum_val += A_i
            
            # Step 3: Calculate the remainder of the current prefix sum
            current_rem = current_prefix_sum_val % modulo
            
            # Step 4: Calculate the target remainder for previous prefix sums (P[l] % modulo)
            # We need (current_rem - target_P_l_rem + modulo) % modulo == k
            # Rearranging for target_P_l_rem: target_P_l_rem = (current_rem - k + modulo) % modulo
            target_P_l_rem = (current_rem - k + modulo) % modulo
            
            # Step 5: Add to total_interesting_subarrays.
            # Each time we find a previous prefix sum P[l] with the target remainder,
            # it means the subarray A[l...i] is interesting.
            total_interesting_subarrays += rem_counts[target_P_l_rem]
            
            # Step 6: Increment the count for the current remainder in rem_counts.
            rem_counts[current_rem] += 1
            
        return total_interesting_subarrays