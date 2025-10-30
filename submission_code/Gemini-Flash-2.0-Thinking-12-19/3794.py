from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)

        # dp[i] will store the finish time of wizard i on the *previous* potion.
        # Initialize with 0s, representing finish times before processing any potion.
        # The finish time of wizard i on potion j is denoted as finish_time[i][j].
        # dp[i] will store finish_time[i][j-1] when we are calculating for potion j.
        dp = [0] * n

        # Iterate through each potion j (from 0 to m-1)
        for j in range(m):
            # When calculating for potion j, dp[i] holds finish_time[i][j-1] for all i.

            # To calculate finish_time[i][j], we need finish_time[i-1][j].
            # We also need finish_time[i+1][j-1] (which is dp[i+1]).

            # Let finish_prev_wizard_current_potion store finish_time[i-1][j] as we iterate through i.
            # For the first wizard (i=0), finish_time[-1][j] is 0.
            finish_prev_wizard_current_potion = 0

            # We need to read the finish times from the previous potion (dp) before they are overwritten
            # when calculating for the current potion j. We need prev_dp[i] and prev_dp[i+1].
            # However, since the update dp[i] = ... uses dp[i+1] which might be from the *previous*
            # state (before processing potion j), we should save the previous state.
            # A simpler way in Python, if updating in place, is to read dp[i+1] from the state
            # *before* the inner loop starts for potion j.

            # Iterate through each wizard i (from 0 to n-1)
            for i in range(n):
                # finish_time[i][j-1] is stored in dp[i] from the previous iteration (or initial 0s).
                # Let's call the value before the inner loop starts old_dp_i and old_dp_i_plus_1.
                # Since dp is updated in place, we need to read the old value before writing the new one.
                # However, dp[i+1] for the constraint check refers to the finish time of wizard i+1
                # on the *previous* potion (j-1). So we need the dp values from the start of the j-loop.

                # Let's use a temporary variable to hold the finish time of wizard i on the previous potion
                # just before it's overwritten.
                prev_finish_i_prev_j = dp[i]

                # The required finish time for wizard i on potion j due to the constraint
                # from wizard i+1 on potion j-1 is finish_time[i+1][j-1], which is the value of dp[i+1]
                # *at the beginning of the current potion's calculation*.
                required_finish_from_next_wiz = dp[i+1] if i < n - 1 else 0 # Read dp[i+1] from the old state

                # Calculate the standard start time for wizard i on potion j.
                # It's the max of when wizard i is free (finished potion j-1) and when potion j arrives at wizard i (wizard i-1 finished potion j).
                wizard_i_free_time = prev_finish_i_prev_j
                potion_j_arrival_time_at_i = finish_prev_wizard_current_potion # This carries the constrained finish time of the previous wizard on the current potion.

                standard_start_time = max(wizard_i_free_time, potion_j_arrival_time_at_i)

                # Calculate the standard finish time for wizard i on potion j, ignoring the constraint from the next wizard for now.
                standard_finish_time = standard_start_time + skill[i] * mana[j]

                # The actual finish time for wizard i on potion j must be at least the standard finish time and also satisfy the constraint from the next wizard.
                curr_finish_i_j = max(standard_finish_time, required_finish_from_next_wiz)

                # Update dp[i] with the calculated finish time for wizard i on the current potion j.
                # This new value will be finish_time[i][j], which becomes finish_time[i][j] for the next iteration i+1
                # (as finish_prev_wizard_current_potion) and finish_time[i][j] for the next potion j+1 (as dp[i]).
                dp[i] = curr_finish_i_j

                # The finish time of the current wizard (i) on the current potion (j) becomes the finish_time[i-1][j] for the next wizard (i+1).
                # This finish_prev_wizard_current_potion must carry the constrained finish time.
                finish_prev_wizard_current_potion = curr_finish_i_j

            # After the inner loop finishes, dp contains finish_time[i][j] for all i.
            # This dp array will be used as the previous potion's finish times (finish_time[i][j])
            # in the next iteration of the outer loop (for potion j+1).

        # After iterating through all potions (j from 0 to m-1),
        # dp[n-1] will contain finish_time[n-1][m-1], which is the minimum total time required.
        return dp[n-1]