import sys
import threading

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    K = int(data[0].strip())
    S = data[1].rstrip('
')
    T = data[2].rstrip('
')

    # Since this problem fixes K = 1, we only need to check if
    # the edit distance between S and T is at most 1.
    # Three cases: equal (0 edits), lengths equal (1 replace),
    # length diff = 1 (1 insert/delete), otherwise >1.

    if S == T:
        print("Yes")
        return

    n, m = len(S), len(T)
    if abs(n - m) > 1:
        print("No")
        return

    # find first position where they differ
    i = 0
    lim = min(n, m)
    while i < lim and S[i] == T[i]:
        i += 1

    # helper to compare the remainder
    def tail_equal(s, si, t, ti):
        # check if s[si:] == t[ti:]
        # manual loop to avoid extra slicing
        while si < len(s) and ti < len(t):
            if s[si] != t[ti]:
                return False
            si += 1
            ti += 1
        # both should reach end together (or it's okay if one is exhausted exactly)
        return si == len(s) and ti == len(t)

    # Case 1: same length -> must be a single replacement
    if n == m:
        # After skipping prefix of length i, the rest must match exactly
        if tail_equal(S, i+1, T, i+1):
            print("Yes")
        else:
            print("No")
        return

    # Case 2: T is longer by exactly 1 -> insert one char into S
    if n + 1 == m:
        # skip T[i], then the tails must match
        if tail_equal(S, i, T, i+1):
            print("Yes")
        else:
            print("No")
        return

    # Case 3: S is longer by exactly 1 -> delete one char from S
    if m + 1 == n:
        # skip S[i], then the tails must match
        if tail_equal(S, i+1, T, i):
            print("Yes")
        else:
            print("No")
        return

    # other cases (shouldn't reach here given abs(n-m)<=1)
    print("No")

if __name__ == "__main__":
    main()