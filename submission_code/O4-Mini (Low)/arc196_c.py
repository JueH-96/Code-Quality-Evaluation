def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    input = sys.stdin.readline
    N = int(input())
    S = input().strip()

    # collect white and black positions
    whites = []
    blacks = []
    for i,ch in enumerate(S, start=1):
        if ch=='W':
            whites.append(i)
        else:
            blacks.append(i)
    # must have exactly N of each
    # sort is automatic since we appended in index order

    # two‐pointer sweep:
    ans = 1
    j = 0        # will scan blacks[0..N-1]
    used = 0     # how many of those we have matched so far
    # we process whites in increasing order of position
    for w in whites:
        # advance j so that blacks[0..j-1] are <= w-1
        while j < N and blacks[j] < w:
            j += 1
        # blacks[0..j-1] are those to the left of w
        free = j - used
        if free <= 0:
            # no available black to match this white so as to cover the cut
            print(0)
            return
        ans = ans * free % MOD
        used += 1

    print(ans)

if __name__ == "__main__":
    main()