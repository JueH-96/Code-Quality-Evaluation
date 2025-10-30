import sys
from collections import defaultdict

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N and M from standard input
        N, M = map(int, sys.stdin.readline().split())
        # Read the list of step counts A
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle empty input or format errors gracefully
        return

    # C[i] will be the cumulative clockwise distance from area 0 to area i.
    # We use 0-based indexing for convenience.
    C = [0] * N
    for i in range(N - 1):
        C[i + 1] = C[i] + A[i]

    # Total distance around the lake.
    total_dist = C[N - 1] + A[N - 1]
    
    # Take all cumulative distances modulo M.
    c_mod = [val % M for val in C]
    total_mod = total_dist % M

    ans = 0
    # `freq` stores the frequency of each remainder encountered so far.
    freq = defaultdict(int)

    # In this loop, `i` represents the starting area index `s`.
    # It first counts pairs where `t < s` (Case 2: s > t) and then updates the frequency map.
    for i in range(N):
        s_mod = c_mod[i]

        # Case 2 (s > t): We are at s=i, and we count valid t's where t < i.
        # The condition is: c_mod[t] == (c_mod[s] - total_mod) % M
        target_t_mod = (s_mod - total_mod + M) % M
        ans += freq[target_t_mod]

        # Update frequency map with the current s_mod to be available for subsequent s's.
        freq[s_mod] += 1

    # After the loop, `freq` contains the final frequency map of all c_mod values.
    # Now, we add the count for Case 1 (s < t).
    # The condition is: c_mod[s] == c_mod[t].
    # For any remainder appearing `k` times, we can form kC2 such pairs.
    for count in freq.values():
        ans += count * (count - 1) // 2

    print(ans)

if __name__ == "__main__":
    main()