# Read K, S, T
# Note: K is always 1 according to the problem statement
# Read K but ignore it based on problem F subproblem description
import sys
# Faster input reading
input = sys.stdin.readline

# K is given as 1 in this sub-problem, but we read it according to format.
K = int(input())

S = input().strip() # Read S and remove trailing newline
T = input().strip() # Read T and remove trailing newline

n = len(S)
m = len(T)

# Case 0: S is already identical to T (0 operations)
if S == T:
    print("Yes")
else:
    # If length difference is more than 1, it's impossible in 1 operation
    if abs(n - m) > 1:
        print("No")
    else:
        # Calculate length of the longest common prefix
        pref = 0
        while pref < min(n, m) and S[pref] == T[pref]:
            pref += 1

        # Calculate length of the longest common suffix that doesn't overlap the prefix
        # We compare characters from the end backwards.
        suff = 0
        while suff < min(n, m): # Max possible suffix length without index going below 0
            s_idx = n - 1 - suff
            t_idx = m - 1 - suff

            # Check if the current suffix indices overlap or touch the prefix region [0, pref - 1]
            # The suffix index must be >= pref
            if s_idx < pref or t_idx < pref:
                break # Suffix has run into/past the prefix region

            if S[s_idx] == T[t_idx]:
                suff += 1
            else:
                break # Mismatch, stop extending suffix

        # Now, analyze based on the relationship between n, m, pref, and suff

        possible = False

        # Case 1a: Potential replacement (n == m, and S != T)
        if n == m:
            # For exactly one replacement, the characters outside
            # the common prefix and suffix must form exactly one pair of
            # differing characters.
            # The number of characters outside the matched prefix and suffix is n - (pref + suff).
            if n - (pref + suff) == 1:
                possible = True

        # Case 1b: Potential insertion into S (m == n + 1)
        elif m == n + 1:
            # For a single insertion to transform S to T, T must be S with one extra character.
            # This means the combined matched prefix and suffix must cover all characters of S (the shorter string).
            # Total characters covered by matches = pref + suff.
            if pref + suff == n:
                possible = True

        # Case 1c: Potential deletion from S (n == m + 1)
        elif n == m + 1:
            # Symmetric to insertion. For a single deletion from S to get T,
            # the combined matched prefix and suffix must cover all characters of T (the shorter string).
            # Total characters covered by matches = pref + suff.
            if pref + suff == m:
                possible = True

        # Output the result
        if possible:
            print("Yes")
        else:
            print("No")