from typing import List

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Helper function to calculate sum(k - current_max_a_upto[k]) for k=1..n
        # This sum is the total number of valid subarrays.
        # For a fixed right endpoint k (1-based), the subarray includes numbers up to k.
        # A subarray ending at k starting at i (1-based) is valid if for all remaining [a, b] with a < b, it's not true that i <= a and b <= k.
        # This means i must be > a for all [a, b] with a < b and b <= k.
        # The minimum valid start i is max({a | [a, b] in remaining, a < b, b <= k} union {0}) + 1.
        # Let max_a_upto[k] = max({a | [a, b] in remaining, a < b, b <= k} union {0}).
        # Minimum valid start is max_a_upto[k] + 1.
        # Valid start indices i are from max_a_upto[k] + 1 up to k.
        # Number of valid start indices = k - (max_a_upto[k] + 1) + 1 = k - max_a_upto[k].
        # Total valid subarrays = sum(k - max_a_upto[k] for k=1..n).
        def calculate_valid_count(n, max_reach_arr):
            total_valid = 0
            current_max_a_upto = 0
            # Iterate through possible right endpoints (1-based index k)
            for k in range(1, n + 1):
                # current_max_a_upto is max({a | [a, b] in S, a < b, b <= k} union {0})
                # which is max(value at k-1, max_reach_arr[k])
                current_max_a_upto = max(current_max_a_upto, max_reach_arr[k])
                # Number of valid subarrays ending at k is k - current_max_a_upto
                total_valid += (k - current_max_a_upto)
            return total_valid

        m = len(conflictingPairs)

        # 1. Compute max_reach_all and second_max_reach_all for the original pairs
        # max_reach_all[k] = max({a | [a, b] in original_pairs, a < b, b == k} union {0})
        # second_max_reach_all[k] = second max such a
        # Initialize with zeros
        max_reach_all = [0] * (n + 1)
        second_max_reach_all = [0] * (n + 1)

        for u, v in conflictingPairs:
            a, b = min(u, v), max(u, v)
            
            # Update max_reach_all and second_max_reach_all for index b
            if a > max_reach_all[b]:
                second_max_reach_all[b] = max_reach_all[b]
                max_reach_all[b] = a
            elif a > second_max_reach_all[b]:
                second_max_reach_all[b] = a

        # 2. Compute current_max_a_upto_all array for k = 1..n
        # current_max_a_upto_all[k] = max({a | [a, b] in original_pairs, a < b, b <= k} union {0})
        # This is the prefix maximum of the max_reach_all array.
        current_max_a_upto_all = [0] * (n + 1)
        current_max = 0
        for k in range(1, n + 1):
            current_max = max(current_max, max_reach_all[k])
            current_max_a_upto_all[k] = current_max

        # 3. Calculate CountValidSubarrays(S_all)
        base_count = calculate_valid_count(n, max_reach_all)
        
        max_valid_count = base_count

        # 4. Iterate through each original pair to consider removing it
        for removed_pair in conflictingPairs:
            u, v = min(removed_pair), max(removed_pair)

            # Only consider removing pairs [u, v] where u was the maximal restriction for v
            # If u was not the maximal restriction, removing it doesn't change max_reach_all[v]
            # and thus doesn't change current_max_a_upto_all or the total count (compared to base_count).
            # The gain from removing such a pair is 0.
            # If u was the maximal restriction, removing it *might* increase the count.
            if u == max_reach_all[v]:
                # This pair is "important" for index v. Removing it lowers max_reach at v.
                
                new_max_reach_v = second_max_reach_all[v]
                
                # Calculate the potential gain in total valid subarrays by removing this pair.
                # Gain = sum_{k=1}^n (new_valid_count_at_k - old_valid_count_at_k)
                # Gain = sum_{k=1}^n ((k - new_current_max_a_upto[k]) - (k - old_current_max_a_upto[k]))
                # Gain = sum_{k=1}^n (old_current_max_a_upto[k] - new_current_max_a_upto[k])
                # The sequences are the same for k < v. So gain is sum_{k=v}^n (...)
                
                potential_gain = 0
                
                # The current max 'a' seen up to k-1 in the new scenario (with pair removed)
                # This value is the same as the old path's current max up to v-1
                # current_max_a_upto_all[v-1] is the prefix max value up to index v-1 (1-based).
                current_new_path_k_minus_1 = current_max_a_upto_all[v-1]

                # Iterate from k = v to n (1-based index) to calculate the difference sum
                k = v
                while k <= n:
                    # Old current max 'a' upto k (from the original set of pairs)
                    old_curr_k_val = current_max_a_upto_all[k]

                    # Max reach value at current k in the new scenario (with pair [u,v] removed)
                    new_max_r_k = max_reach_all[k] # Default value for k != v
                    if k == v:
                         new_max_r_k = new_max_reach_v

                    # New current max 'a' upto k
                    current_new_path_k = max(current_new_path_k_minus_1, new_max_r_k)
                    
                    # Gain at index k is the reduction in max_a_upto value
                    gain_at_k = old_curr_k_val - current_new_path_k
                    
                    # The difference old_curr_k_val - current_new_path_k should always be >= 0.
                    # It represents the increase in the number of valid subarrays ending at k.
                    potential_gain += gain_at_k

                    # Optimization: If the new path has caught up to the old path at index k,
                    # meaning current_new_path_k == old_curr_k_val, then for all subsequent indices j > k,
                    # the new path current_max_a_upto_new[j] will be equal to the old path
                    # current_max_a_upto_all[j]. This is because the max_reach values are the same
                    # for j > v, and both sequences are prefix maximums.
                    # Once A_k = B_k, and M_j = N_j for j > k, then A_j = max(A_{j-1}, M_j) and B_j = max(B_{j-1}, M_j).
                    # Since A_{j-1} >= B_{j-1}, max(A_{j-1}, M_j) >= max(B_{j-1}, M_j).
                    # If A_k = B_k, then A_{k+1} = max(A_k, M_{k+1}) and B_{k+1} = max(B_k, M_{k+1}). They must be equal.
                    # By induction, A_j = B_j for all j > k.
                    # So the gain from k+1 to n is 0. We can break the loop.
                    if current_new_path_k == old_curr_k_val:
                         break # Remaining gain from k+1 to n is 0.
                    
                    # Move to the next index
                    k += 1
                    # Update the current_new_path_k_minus_1 for the next iteration
                    current_new_path_k_minus_1 = current_new_path_k 

                # Update the maximum valid count found so far by considering the gain from removing this pair.
                max_valid_count = max(max_valid_count, base_count + potential_gain)

        return max_valid_count