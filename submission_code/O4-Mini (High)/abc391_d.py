import sys
import threading
def main():
    import sys
    input = sys.stdin.readline

    N, W = map(int, input().split())
    # Group blocks by column
    cols = [[] for _ in range(W + 1)]
    for i in range(1, N + 1):
        x, y = map(int, input().split())
        cols[x].append((y, i))

    # Compute m = minimum number of blocks in any column
    m = None
    for x in range(1, W + 1):
        k = len(cols[x])
        if k == 0:
            m = 0
            break
        if m is None or k < m:
            m = k

    Q = int(input())
    # If some column is empty, no full clears ever happen => no block is removed
    if m == 0:
        # just read queries and answer Yes
        for _ in range(Q):
            input()
            sys.stdout.write("Yes
")
        return

    # M[s] = time when the s-th block in each column is cleared:
    # M[s] = max over columns of initial Y of the s-th block in that column
    M = [0] * (m + 1)   # 1-based up to m
    rank = [0] * (N + 1)  # rank[i] = position in its column stack

    # Sort each column by height and record ranks and update M
    for x in range(1, W + 1):
        lst = cols[x]
        lst.sort()  # sort by y ascending
        # assign ranks
        for idx, (y, idx_block) in enumerate(lst):
            s = idx + 1
            rank[idx_block] = s
            if s <= m and y > M[s]:
                M[s] = y

    # Compute removal time for each block
    INF = 10**18
    removal_time = [INF] * (N + 1)
    for i in range(1, N + 1):
        s = rank[i]
        if 1 <= s <= m:
            removal_time[i] = M[s]
        # else remains INF

    out = []
    # Answer queries
    for _ in range(Q):
        line = input().split()
        T = int(line[0]); A = int(line[1])
        # block A exists at time T+0.5 iff T < removal_time[A]
        if T < removal_time[A]:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()