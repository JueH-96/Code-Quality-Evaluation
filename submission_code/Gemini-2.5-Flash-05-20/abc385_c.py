import sys

def solve():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))

    # If N is 0 or 1, the maximum number of buildings is N itself.
    # The problem constraints state 1 <= N, so N=0 is not a case.
    if N <= 1:
        sys.stdout.write(str(N) + "
")
        return

    # dp[i][j] will store the maximum length of an arithmetic progression
    # whose last element is H[i] (at index i) and whose second-to-last element
    # is H[j] (at index j).
    # Initialize all values to 1, representing a single building (base case for any AP).
    dp = [[1] * N for _ in range(N)]

    # Initialize max_ans to 1, as choosing a single building is always possible.
    max_ans = 1

    # Iterate over all possible last buildings (index i)
    for i in range(N):
        # Iterate over all possible second-to-last buildings (index j)
        # j must be less than i to be a previous element.
        for j in range(i):
            # Condition 1: The chosen buildings must have the same height.
            if H[i] == H[j]:
                # If H[i] and H[j] are the same, they form an AP of length 2.
                dp[i][j] = 2

                # Condition 2: The chosen buildings must be arranged at equal intervals.
                # Calculate the common difference based on i and j.
                d = i - j
                
                # Calculate the index of the building that would come before H[j] in this AP.
                prev_prev_idx = j - d

                # Check if prev_prev_idx is a valid index (non-negative).
                if prev_prev_idx >= 0:
                    # Check if the height of the building at prev_prev_idx also matches
                    # the common height of H[j] (and H[i]).
                    if H[prev_prev_idx] == H[j]:
                        # If it matches, we can extend the AP ending at j with prev_prev_idx.
                        # dp[j][prev_prev_idx] holds the length of the AP up to j.
                        # Adding H[i] extends it by 1.
                        dp[i][j] = max(dp[i][j], dp[j][prev_prev_idx] + 1)
            
            # Update the overall maximum length found so far.
            max_ans = max(max_ans, dp[i][j])
    
    sys.stdout.write(str(max_ans) + "
")

solve()