import sys

def solve():
    """
    This function solves the problem using dynamic programming.
    It calculates the minimum cost to transform a given binary string S into a "good string".
    A string is "good" if it has exactly one pair of adjacent identical characters.
    
    The approach is as follows:
    1. Define two DP tables:
       - `dp_pref[k][c]`: Min cost to make prefix S[0...k] alternate and end with char `c`.
       - `dp_suff[k][c]`: Min cost to make suffix S[k...N-1] alternate and start with char `c`.
    
    2. Compute `dp_pref` in a forward pass and `dp_suff` in a backward pass.
       - The cost to change S[k] to 0 is `C[k]` if S[k] is 1, and 0 otherwise. This can be expressed
         as `C[k] * S_int[k]` if S is converted to a list of integers.
       - The cost to change S[k] to 1 is `C[k] * (1 - S_int[k])`.
    
    3. A good string T with T[i] == T[i+1] is formed by an alternating prefix ending at `i` and an
       alternating suffix starting at `i+1`.
    
    4. Iterate through all possible split points `i` from 0 to N-2. For each `i`, calculate the cost
       for the repeated pair to be `00` or `11`.
       - Cost for `00` at (i, i+1): `dp_pref[i][0] + dp_suff[i+1][0]`
       - Cost for `11` at (i, i+1): `dp_pref[i][1] + dp_suff[i+1][1]`
    
    5. The minimum of these costs over all `i` is the answer.
    """
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))

    # Convert string '0'/'1' to integers 0/1 for easier calculations
    S = [int(c) for c in S_str]

    # dp_pref[k][c]: min cost to make S[0..k] an alternating string ending in char c
    dp_pref = [[0] * 2 for _ in range(N)]
    
    # Base case k=0
    # Cost to make S[0] into 0
    dp_pref[0][0] = C[0] * S[0]
    # Cost to make S[0] into 1
    dp_pref[0][1] = C[0] * (1 - S[0])

    # Fill dp_pref for k > 0
    for k in range(1, N):
        # To end with '0', previous must end with '1'
        dp_pref[k][0] = dp_pref[k-1][1] + C[k] * S[k]
        # To end with '1', previous must end with '0'
        dp_pref[k][1] = dp_pref[k-1][0] + C[k] * (1 - S[k])

    # dp_suff[k][c]: min cost to make S[k..N-1] an alternating string starting with char c
    dp_suff = [[0] * 2 for _ in range(N)]
    
    # Base case k=N-1
    # Cost to make S[N-1] into 0
    dp_suff[N-1][0] = C[N-1] * S[N-1]
    # Cost to make S[N-1] into 1
    dp_suff[N-1][1] = C[N-1] * (1 - S[N-1])

    # Fill dp_suff for k < N-1 (from right to left)
    for k in range(N - 2, -1, -1):
        # To start with '0', next must start with '1'
        dp_suff[k][0] = dp_suff[k+1][1] + C[k] * S[k]
        # To start with '1', next must start with '0'
        dp_suff[k][1] = dp_suff[k+1][0] + C[k] * (1 - S[k])

    min_cost = float('inf')

    # Iterate through all possible positions `i` for the repeated pair T[i] == T[i+1]
    for i in range(N - 1):
        # Case 1: T[i] = T[i+1] = 0
        cost0 = dp_pref[i][0] + dp_suff[i+1][0]
        min_cost = min(min_cost, cost0)

        # Case 2: T[i] = T[i+1] = 1
        cost1 = dp_pref[i][1] + dp_suff[i+1][1]
        min_cost = min(min_cost, cost1)

    print(min_cost)

if __name__ == "__main__":
    solve()