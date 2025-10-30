from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        m = len(mana)
        # dp[j] will be the finish time of potion j after processing by the current wizard
        dp = [0] * m
        for s in skill:
            # Process the 0th potion for current wizard
            dp[0] = dp[0] + s * mana[0]
            # For every subsequent potion j, the start time must be no earlier than:
            #   - the finish time of potion j with previous wizard (dp[j])
            #   - the finish time of potion (j-1) with current wizard (dp[j-1])
            for j in range(1, m):
                dp[j] = max(dp[j - 1], dp[j]) + s * mana[j]
        return dp[-1]


###############################################################################
# The following code is provided to allow testing from standard input.
# Input is expected as two lines:
#   Line 1: the integers for skill, separated by spaces (or commas)
#   Line 2: the integers for mana, separated by spaces (or commas)
###############################################################################
def solve():
    import sys
    input_data = sys.stdin.read().strip().splitlines()
    if not input_data:
        return

    # Function to parse a line into a list of integers
    def parse_line(line: str) -> List[int]:
        # Remove unwanted characters and then split by spaces
        for ch in "[],=":
            line = line.replace(ch, " ")
        return list(map(int, line.split()))
    
    # We assume that input is provided in two lines:
    # one for skill and one for mana.
    if len(input_data) >= 2:
        skill = parse_line(input_data[0])
        mana = parse_line(input_data[1])
    else:
        # Fallback: if input is provided on one line, try to split it.
        tokens = parse_line(input_data[0])
        mid = len(tokens) // 2
        skill = tokens[:mid]
        mana = tokens[mid:]
    
    sol = Solution()
    result = sol.minTime(skill, mana)
    sys.stdout.write(str(result))

if __name__ == '__main__':
    solve()