import sys

# Set a higher recursion limit for the reconstruction phase on large inputs.
sys.setrecursionlimit(50005)

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)

        # A good caption is composed of groups of length >= 3.
        # It's impossible to form such a string if n is 1 or 2.
        if n > 0 and n < 3:
            return ""
        if n == 0:
            return ""

        ords = [ord(c) for c in caption]

        # Precompute prefix sums for character counts and costs.
        # counts[i][c]: count of character 'a'+c in caption[:i]
        # costs[i][c]: cost to change caption[:i] to all 'a'+c
        counts = [[0] * 26 for _ in range(n + 1)]
        costs = [[0] * 26 for _ in range(n + 1)]

        for i in range(n):
            for char_code in range(26):
                char_ord = ord('a') + char_code
                counts[i + 1][char_code] = counts[i][char_code]
                costs[i + 1][char_code] = costs[i][char_code] + abs(ords[i] - char_ord)
            
            counts[i + 1][ords[i] - ord('a')] += 1

        memo_median_cost = {}
        def get_median_and_cost(j: int, i: int):
            if (j, i) in memo_median_cost:
                return memo_median_cost[(j, i)]

            length = i - j
            
            substr_counts = [counts[i][c] - counts[j][c] for c in range(26)]
            
            # To find the median that minimizes cost and string lexicographically,
            # we find the character at index (length-1)//2 of the sorted characters.
            # This is equivalent to finding the ( (length+1)//2 )-th character.
            count_so_far = 0
            median_code = -1
            target_count = (length + 1) // 2
            
            for c_code in range(26):
                count_so_far += substr_counts[c_code]
                if count_so_far >= target_count:
                    median_code = c_code
                    break
            
            cost = costs[i][median_code] - costs[j][median_code]
            
            result = (chr(ord('a') + median_code), cost)
            memo_median_cost[(j, i)] = result
            return result

        # dp[i] = (cost, list_of_optimal_choices)
        # A choice is (parent_j, char_for_last_segment)
        dp = [(float('inf'), []) for _ in range(n + 1)]
        dp[0] = (0, [])
        
        # Fill the DP table
        for i in range(1, n + 1):
            if i < 3:
                continue
            for k in range(3, i + 1):
                j = i - k
                if dp[j][0] == float('inf'):
                    continue
                
                median_char, segment_cost = get_median_and_cost(j, i)
                new_cost = dp[j][0] + segment_cost
                
                if new_cost < dp[i][0]:
                    dp[i] = (new_cost, [(j, median_char)])
                elif new_cost == dp[i][0]:
                    dp[i][1].append((j, median_char))

        if dp[n][0] == float('inf'):
            return ""
        
        # Reconstruct the lexicographically smallest string
        memo_reconstruct = {}
        def reconstruct(i: int) -> str:
            if i == 0:
                return ""
            if i in memo_reconstruct:
                return memo_reconstruct[i]

            _cost, choices = dp[i]
            # Use a sentinel value that is lexicographically larger than any possible result
            best_str = "~" 
            
            for j, char in choices:
                k = i - j
                candidate_str = reconstruct(j) + char * k
                if best_str == "~" or candidate_str < best_str:
                    best_str = candidate_str

            memo_reconstruct[i] = best_str
            return best_str

        return reconstruct(n)