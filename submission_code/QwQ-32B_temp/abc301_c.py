import sys
from collections import defaultdict

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

    counts_S = defaultdict(int)
    counts_T = defaultdict(int)

    for c in S:
        if c != '@':
            counts_S[c] += 1
    for c in T:
        if c != '@':
            counts_T[c] += 1

    # Check non-allowed characters
    all_chars = set(counts_S.keys()).union(set(counts_T.keys()))
    for c in all_chars:
        if c not in allowed and c != '@':
            if counts_S.get(c, 0) != counts_T.get(c, 0):
                print("No")
                return

    # Calculate D for each allowed character
    D = {}
    for c in allowed:
        D[c] = counts_T.get(c, 0) - counts_S.get(c, 0)

    min_sum = 0
    for c in allowed:
        min_a = max(D[c], 0)
        min_sum += min_a

    s_at = S.count('@')
    if min_sum > s_at:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()