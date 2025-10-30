import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)

    # read input
    N, K = map(int, sys.stdin.readline().split())
    lost_colors = set(map(int, sys.stdin.readline().split()))

    # build the (already sorted) list of remaining socks
    socks = []
    for c in range(1, N + 1):
        socks.append(c)           # the remaining sock of this colour
        if c not in lost_colors:  # if the pair is still complete, add the second one
            socks.append(c)

    M = len(socks)               # total number of remaining socks

    # Case 1 : even number of socks  -> just pair neighbouring socks
    if M % 2 == 0:
        answer = 0
        for i in range(0, M, 2):
            answer += socks[i + 1] - socks[i]      # socks is sorted ⇒ abs = difference
        print(answer)
        return

    # Case 2 : odd number of socks  -> one sock will stay unpaired
    # left[i]  : minimum cost to pair the first i socks (i must be even)
    left = [0] * (M + 1)
    for i in range(2, M + 1, 2):
        left[i] = left[i - 2] + (socks[i - 1] - socks[i - 2])

    # right[i] : minimum cost to pair the socks from index i to the end (length even)
    right = [0] * (M + 2)         # two extra cells so that i+2 is always safe
    right[M] = 0
    for i in range(M - 2, -1, -1):
        if (M - i) % 2 == 0:      # the remaining length is even ⇒ can be fully paired
            right[i] = (socks[i + 1] - socks[i]) + right[i + 2]

    # try every possible choice of the sock that is left alone
    # (only indices with even prefix length need to be considered, i.e. r even)
    INF = 10 ** 18
    ans = INF
    for r in range(0, M, 2):      # r is the index of the unused sock
        ans = min(ans, left[r] + right[r + 1])

    print(ans)

if __name__ == "__main__":
    main()