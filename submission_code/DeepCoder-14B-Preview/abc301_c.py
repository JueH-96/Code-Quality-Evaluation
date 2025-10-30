def solve():
    import sys
    from collections import defaultdict

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

    # Check for any non-allowed, non-@ characters with differing counts
    all_chars = set(S) | set(T)
    for c in all_chars:
        if c not in allowed and c != '@':
            if S.count(c) != T.count(c):
                print("No")
                return

    # Count allowed characters and '@' in both S and T
    count_S = defaultdict(int)
    for c in S:
        if c in allowed or c == '@':
            count_S[c] += 1

    count_T = defaultdict(int)
    for c in T:
        if c in allowed or c == '@':
            count_T[c] += 1

    # Calculate sum of allowed characters in S and T
    sum_S_allowed = sum(count_S[x] for x in allowed)
    sum_T_allowed = sum(count_T[x] for x in allowed)

    S_at = count_S.get('@', 0)
    T_at = count_T.get('@', 0)

    # Check if the difference in allowed sums can be balanced by @ replacements
    if (sum_T_allowed - sum_S_allowed) != (S_at - T_at):
        print("No")
        return

    # Calculate delta for each allowed character
    delta = {}
    for x in allowed:
        delta[x] = count_T.get(x, 0) - count_S.get(x, 0)

    # Sum of minimal required s_x (delta_x where delta_x > 0)
    sum_min_sx = sum(dx for dx in delta.values() if dx > 0)

    if sum_min_sx > S_at:
        print("No")
        return

    print("Yes")

solve()