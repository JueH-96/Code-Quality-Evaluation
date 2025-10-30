import sys
from math import comb

def main() -> None:
    sys.setrecursionlimit(10000)

    # read input
    N, M = map(int, sys.stdin.readline().split())

    # K is the maximum value the transformed sequence B_i can take
    # where B_i = A_i - 10*(i-1)
    K = M - 10 * (N - 1)

    # number of weakly–increasing sequences of length N
    # whose elements are in [1, K]  ->  C(N + K - 1, N)
    total = comb(N + K - 1, N)
    print(total)                       # first line: X

    seq = []                           # current sequence under construction

    # depth-first search that enumerates the sequences in lexicographical order
    def dfs(pos: int, prev_val: int) -> None:
        if pos == N:                   # finished one valid sequence
            sys.stdout.write(' '.join(map(str, seq)) + '
')
            return

        # smallest and largest value we may put at position `pos`
        min_val = 1 if pos == 0 else prev_val + 10
        max_val = M - 10 * (N - pos - 1)   # must leave room for the rest

        for v in range(min_val, max_val + 1):
            seq.append(v)
            dfs(pos + 1, v)
            seq.pop()

    dfs(0, 0)

if __name__ == "__main__":
    main()