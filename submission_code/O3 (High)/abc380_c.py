import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)

    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # collect all 1-blocks (0-based indices)
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1':
            l = i
            while i < N and S[i] == '1':
                i += 1
            r = i - 1
            blocks.append((l, r))
        else:
            i += 1

    # indices of the (K-1)-th and K-th blocks (0-based counting in the list)
    prev_l, prev_r = blocks[K - 2]
    cur_l, cur_r = blocks[K - 1]
    length = cur_r - cur_l + 1          # length of the K-th block

    res = list(S)

    # remove the K-th block from its original position
    for pos in range(cur_l, cur_r + 1):
        res[pos] = '0'

    # insert it right after the previous block
    insert_start = prev_r + 1
    for pos in range(insert_start, insert_start + length):
        res[pos] = '1'

    print(''.join(res))


if __name__ == "__main__":
    main()