from typing import List
from collections import Counter
from bisect import bisect_right

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # 1. Count frequencies of each damage value.
        # This allows us to group spells by damage and calculate total damage per unique value.
        counts = Counter(power)

        # 2. Get unique damage values and sort them.
        # Sorting is crucial for the dynamic programming approach and using bisect_right.
        unique_powers = sorted(counts.keys())
        m = len(unique_powers)

        # If there are no unique powers (e.g., input list is empty), total damage is 0.
        if m == 0:
            return 0

        # 3. Compute the total damage available for each unique value.
        # This is the sum of damages from all spells with that specific power.
        # Store them in a list `freq_values` which aligns with `unique_powers`.
        freq_values = [unique_powers[i] * counts[unique_powers[i]] for i in range(m)]

        # 4. Initialize DP array.
        # dp[i] will store the maximum possible total damage considering only spells
        # with damages from `unique_powers[0]` up to `unique_powers[i]`.
        dp = [0] * m

        # 5. Iterate and fill the DP array.
        # Base case for the first unique power: We can either take all spells with this damage
        # or take none. Taking all gives freq_values[0], taking none gives 0. The maximum is freq_values[0].
        dp[0] = freq_values[0]

        # Iterate from the second unique power onwards.
        for i in range(1, m):
            current_power = unique_powers[i]
            current_damage = freq_values[i]

            # Option 1: Do not cast any spell with the current_power (unique_powers[i]).
            # In this case, the maximum damage is the same as the maximum damage considering
            # unique powers up to unique_powers[i-1].
            option1 = dp[i - 1]

            # Option 2: Cast all spells with the current_power (unique_powers[i]).
            # The damage gained is current_damage.
            # According to the rule, if we cast a spell with damage D, we cannot cast
            # any spell with damage D' if |D - D'| is 1 or 2.
            # If we choose current_power (unique_powers[i]), we cannot choose any unique power u_j
            # (where j < i) if abs(unique_powers[i] - u_j) is 1 or 2.
            # Since u_j < unique_powers[i] for j < i, this condition simplifies to
            # unique_powers[i] - u_j is 1 or 2.
            # Thus, we can only combine current_power with previous unique powers u_j where
            # unique_powers[i] - u_j is NOT 1 or 2. Since it must be positive, this means
            # unique_powers[i] - u_j >= 3, or u_j <= unique_powers[i] - 3.

            # We need to find the maximum damage from a valid subset of unique powers from
            # unique_powers[0]...unique_powers[i-1] such that every selected power u_j satisfies
            # u_j <= unique_powers[i] - 3.
            # This is equivalent to finding the maximum damage from a valid subset of
            # unique_powers[0]...unique_powers[p], where p is the largest index j < i
            # such that unique_powers[j] <= unique_powers[i] - 3.

            # We use bisect_right to find the insertion point `k` for `search_value = current_power - 3`
            # in the sorted list `unique_powers[:i]`. `bisect_right` finds the index `k` such that
            # elements `unique_powers[0...k-1]` are all less than or equal to `search_value`,
            # and elements `unique_powers[k...i-1]` are all greater than `search_value`.
            # The largest index `p` such that unique_powers[p] <= search_value is `k - 1`.

            search_value = current_power - 3
            
            # Find the index k in unique_powers[:i]
            # hi=i ensures search is limited to the previous elements.
            k = bisect_right(unique_powers, search_value, hi=i)

            # The index of the last element <= search_value is k - 1.
            # This is the index 'p' in our logic.
            prev_j_for_dp = k - 1

            # The maximum damage from a valid subset using unique powers up to index p is dp[p].
            # If p is negative (meaning no unique power <= search_value exists before index i),
            # the previous maximum damage is 0.
            prev_dp_value = dp[prev_j_for_dp] if prev_j_for_dp >= 0 else 0

            # Option 2 is the current damage from taking unique_powers[i] plus the maximum
            # damage from compatible previous unique powers.
            option2 = current_damage + prev_dp_value

            # dp[i] is the maximum of the two choices: either don't take unique_powers[i]
            # or take unique_powers[i].
            dp[i] = max(option1, option2)

        # 6. The final result is dp[m-1], representing the maximum total damage
        # considering all unique powers.
        return dp[m - 1]