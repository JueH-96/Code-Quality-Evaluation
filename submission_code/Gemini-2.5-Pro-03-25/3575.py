import collections
from typing import List

class Solution:
    """
    Solves the problem using dynamic programming.
    The state `dp[(j, p)]` stores information about achievable OR sums A and B 
    using 'j' elements for set S1 and 'p' elements for set S2, chosen disjointly 
    from the elements processed so far from `nums`.

    Specifically, `dp[(j, p)]` is a list (or array) of size `V=128`. 
    The element `dp[(j, p)][a]` is an integer acting as a bitmask. 
    If the b-th bit of `dp[(j, p)][a]` is set, it means it's possible to achieve 
    an OR sum of `A=a` using `j` elements in S1 and an OR sum of `B=b` using `p` 
    elements in S2.

    V = 128 because the maximum possible OR sum from numbers less than 2^7 (128) is 127. 
    So the OR sums A and B will be in the range [0, 127].
    """
    def maxValue(self, nums: List[int], k: int) -> int:
        
        V = 128 # Range of possible OR sums is [0, 127]. V=128 covers indices 0 to 127.

        # Initialize the DP table. Use defaultdict for potentially sparse storage.
        # Key: tuple (j, p) representing counts for S1 and S2.
        # Value: A list of size V, where index 'a' stores the bitmask for possible B values.
        dp = collections.defaultdict(lambda: [0] * V)
        
        # Base case: With 0 elements chosen for S1 (j=0) and 0 elements for S2 (p=0),
        # the only possible OR sums are A=0 and B=0.
        # We represent the set of possible B values {0} with the bitmask 1 (which is 1 << 0).
        dp[0, 0][0] = 1 

        # Iterate through each number in the input list `nums`.
        for num in nums:
            # Create a temporary dictionary `updates` to store the changes/new states 
            # resulting from including the current `num`. This avoids modifying `dp` while iterating.
            updates = collections.defaultdict(lambda: [0] * V)

            # Iterate through all existing reachable states (j, p) in the current `dp` table.
            # list(dp.items()) creates a snapshot to iterate over safely.
            for state_key, masks_for_A in list(dp.items()):
                j, p = state_key

                # Consider adding the current `num` to set S1.
                # This transition is possible only if j < k.
                # It affects the state (j+1, p). The calculation uses info from state (j, p).
                if j < k:
                    target_state_key = (j + 1, p)
                    # Retrieve the list of masks for the target state in `updates` dictionary.
                    # If the key doesn't exist, defaultdict creates a new list [0]*V.
                    target_masks = updates[target_state_key] 
                    # Iterate through all possible OR sums A=a_prev for the current state (j, p).
                    for a_prev in range(V):
                        # Check if A=a_prev was possible (mask is non-zero).
                        if masks_for_A[a_prev] != 0: 
                            # Calculate the new OR sum for S1.
                            new_a = a_prev | num
                            # Ensure the new OR sum A is within bounds [0, V-1].
                            if new_a < V:
                                # The set of possible B values remains the same as for state (j, p) with A=a_prev.
                                # We merge these possibilities into the target state (j+1, p) for A=new_a.
                                # Using bitwise OR merges the sets of B values represented by the masks.
                                target_masks[new_a] |= masks_for_A[a_prev]

                # Consider adding the current `num` to set S2.
                # This transition is possible only if p < k.
                # It affects the state (j, p+1). The calculation uses info from state (j, p).
                if p < k:
                    target_state_key = (j, p + 1)
                    # Retrieve the list of masks for the target state in `updates`.
                    target_masks = updates[target_state_key] 
                    # Iterate through all possible OR sums A=a for the current state (j, p).
                    for a in range(V):
                        mask_B_prev = masks_for_A[a]
                        # Check if there were any possible B values for A=a in state (j, p).
                        if mask_B_prev != 0:
                            # Calculate the new mask of B values. For each B=b present in `mask_B_prev`,
                            # the new B value will be `b | num`. We collect these new B values into `new_mask_B`.
                            new_mask_B = 0
                            # Iterate through the bits set in mask_B_prev efficiently.
                            temp_mask = mask_B_prev
                            while temp_mask > 0:
                                # Isolate the lowest set bit value (this will be a power of 2).
                                b_val = temp_mask & -temp_mask 
                                # Calculate the index `b_idx` corresponding to this bit value.
                                # Example: if b_val=4 (binary 100), b_idx=2.
                                b_idx = b_val.bit_length() - 1
                                
                                # Calculate the new OR sum B.
                                new_b_idx = b_idx | num
                                # Ensure the new OR sum B is within bounds [0, V-1].
                                if new_b_idx < V:
                                    # Set the corresponding bit in the new mask for B values.
                                    new_mask_B |= (1 << new_b_idx)
                                
                                # Remove the processed lowest set bit from temp_mask to continue iteration.
                                temp_mask ^= b_val
                            
                            # Merge the newly calculated possible B values (`new_mask_B`) into the 
                            # target state (j, p+1)'s mask list at index `a`.
                            target_masks[a] |= new_mask_B

            # After calculating all updates based on the current `num`, merge them into the main `dp` table.
            # The states achievable by *skipping* `num` are already present in `dp`.
            # We only need to incorporate the new possibilities generated by using `num`.
            for state_key, updated_masks in updates.items():
                # Get the list of masks for the state in `dp`. If the state `state_key` is new, 
                # defaultdict creates a new list [0]*V.
                current_masks = dp[state_key] 
                # Iterate through all possible A values (indices 0 to V-1).
                for a in range(V):
                    # Merge the updates using bitwise OR.
                    current_masks[a] |= updated_masks[a]

        # After processing all numbers in `nums`, find the maximum A XOR B value.
        # This value must come from the final state where j=k and p=k.
        max_xor_val = 0
        # Check if the target state (k, k) was ever reached.
        if (k, k) in dp:
            # Retrieve the list of masks for the final state (k, k).
            final_masks_for_A = dp[k, k]
            # Iterate through all possible OR sums A=a.
            for a in range(V):
                mask_B = final_masks_for_A[a]
                # Check if there are any possible B values associated with this A=a.
                if mask_B != 0:
                    # Iterate through all B values represented in the `mask_B`.
                    temp_mask = mask_B
                    while temp_mask > 0:
                        b_val = temp_mask & -temp_mask
                        b_idx = b_val.bit_length() - 1
                        
                        # Calculate the XOR value for the current pair (A=a, B=b_idx).
                        # Update the maximum XOR value found so far.
                        max_xor_val = max(max_xor_val, a ^ b_idx)
                        
                        # Remove the processed B value bit.
                        temp_mask ^= b_val
        
        # Return the overall maximum XOR value found.
        return max_xor_val