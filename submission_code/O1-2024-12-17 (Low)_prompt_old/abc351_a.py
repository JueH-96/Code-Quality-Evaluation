def solve():
    import sys
    # Read inputs
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Calculate the total runs of Takahashi after top of the 9th
    score_takahashi = sum(A)
    # Calculate the total runs of Aoki after bottom of the 8th
    score_aoki = sum(B)

    # To win, Aoki needs strictly more runs than Takahashi.
    # Minimum needed in the 9th = (Takahashi_score - Aoki_score) + 1
    runs_needed = (score_takahashi - score_aoki) + 1

    print(runs_needed)

# Call solve() to execute
if __name__ == "__main__":
    solve()