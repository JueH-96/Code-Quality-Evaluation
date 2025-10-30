import sys

# Use a large value for infinity
INF = float('inf')

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)

        if n < 3:
            # A good caption must have blocks of length at least 3.
            # The total length must be at least 3.
            return ""

        # dp[i][char_code][length]
        # i: index 0 to n-1
        # char_code: 0 to 25 ('a' to 'z')
        # length: 1, 2, 3 (where 3 means >= 3)
        dp = [[[INF] * 4 for _ in range(26)] for _ in range(n)]

        # Base case i = 0
        for char_code in range(26):
            cost = abs(ord(caption[0]) - ord('a') - char_code)
            dp[0][char_code][1] = cost

        # Fill DP table
        for i in range(1, n):
            current_char_code_original = ord(caption[i]) - ord('a')

            for current_char_code in range(26):
                cost = abs(current_char_code_original - current_char_code)

                # Calculate dp[i][current_char_code][1]
                # Comes from dp[i-1][prev_char_code][3] where prev_char_code != current_char_code
                min_prev_cost_len3 = INF
                for prev_char_code in range(26):
                    if prev_char_code != current_char_code:
                         min_prev_cost_len3 = min(min_prev_cost_len3, dp[i-1][prev_char_code][3])

                if min_prev_cost_len3 != INF:
                    dp[i][current_char_code][1] = min_prev_cost_len3 + cost

                # Calculate dp[i][current_char_code][2]
                # Comes from dp[i-1][current_char_code][1]
                if dp[i-1][current_char_code][1] != INF:
                    dp[i][current_char_code][2] = dp[i-1][current_char_code][1] + cost

                # Calculate dp[i][current_char_code][3]
                # Comes from dp[i-1][current_char_code][2] or dp[i-1][current_char_code][3]
                min_prev_cost_len2or3 = INF
                min_prev_cost_len2or3 = min(min_prev_cost_len2or3, dp[i-1][current_char_code][2])
                min_prev_cost_len2or3 = min(min_prev_cost_len2or3, dp[i-1][current_char_code][3])


                if min_prev_cost_len2or3 != INF:
                    dp[i][current_char_code][3] = min_prev_cost_len2or3 + cost


        # Find the minimum total cost at the end (must end in a block of length >= 3)
        min_total_cost = INF
        end_char_code = -1

        for char_code in range(26):
            if dp[n-1][char_code][3] < min_total_cost:
                min_total_cost = dp[n-1][char_code][3]
                end_char_code = char_code

        # If minimum cost is infinity, it's impossible
        if min_total_cost == INF:
            return ""

        # Reconstruct the lexicographically smallest string
        result = [""] * n
        current_char_code = end_char_code
        current_length = 3 # The optimal solution must end in a block of length >= 3

        for i in range(n - 1, -1, -1):
            result[i] = chr(current_char_code + ord('a'))

            if i == 0:
                break # Reached the beginning

            current_cost_at_i = abs(ord(caption[i]) - ord('a') - current_char_code)
            target_prev_cost = dp[i][current_char_code][current_length] - current_cost_at_i

            found_prev_state = False

            # Try to find the previous state (prev_char_code, prev_length)
            # Iterate through prev_char_code from 0 to 25 to get lexicographically smallest result[i-1]
            for prev_char_code_candidate in range(26):

                # Check valid transitions based on current_length and prev_char_code_candidate

                # Case 1: Current state is length 1 (current_length=1)
                # Must come from previous state (i-1, prev_char_code_candidate, prev_length=3)
                # where prev_char_code_candidate != current_char_code
                if current_length == 1 and prev_char_code_candidate != current_char_code:
                    required_prev_length = 3
                    if dp[i-1][prev_char_code_candidate][required_prev_length] == target_prev_cost:
                        current_char_code = prev_char_code_candidate
                        current_length = required_prev_length
                        found_prev_state = True
                        break # Found the prev_char_code, break from this loop

                # Case 2: Current state is length 2 (current_length=2)
                # Must come from previous state (i-1, prev_char_code_candidate, prev_length=1)
                # where prev_char_code_candidate == current_char_code
                elif current_length == 2 and prev_char_code_candidate == current_char_code:
                     required_prev_length = 1
                     if dp[i-1][prev_char_code_candidate][required_prev_length] == target_prev_cost:
                         current_char_code = prev_char_code_candidate # which is current_char_code
                         current_length = required_prev_length # which is 1
                         found_prev_state = True
                         break # Found the prev_char_code (must be current_char_code), break

                # Case 3: Current state is length 3 (current_length=3)
                # Must come from previous state (i-1, prev_char_code_candidate, prev_length=2 or 3)
                # where prev_char_code_candidate == current_char_code
                elif current_length == 3 and prev_char_code_candidate == current_char_code:
                    # Check prev_length=2 first
                    required_prev_length_option1 = 2
                    if dp[i-1][prev_char_code_candidate][required_prev_length_option1] == target_prev_cost:
                         current_char_code = prev_char_code_candidate
                         current_length = required_prev_length_option1
                         found_prev_state = True
                         break # Found the prev_char_code, break

                    # If prev_length=2 didn't match cost, check prev_length=3
                    required_prev_length_option2 = 3
                    if dp[i-1][prev_char_code_candidate][required_prev_length_option2] == target_prev_cost:
                         current_char_code = prev_char_code_candidate
                         current_length = required_prev_length_option2
                         found_prev_state = True
                         break # Found the prev_char_code, break


        return "".join(result)