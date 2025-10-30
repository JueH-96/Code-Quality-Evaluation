import sys
from array import array

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    H = list(map(int, data[1:]))

    # collect positions of every height (1-indexed positions)
    pos_of_height = {}
    for idx, h in enumerate(H, 1):
        pos_of_height.setdefault(h, []).append(idx)

    best = 1  # at least one building can always be chosen

    for pos_list in pos_of_height.values():
        k = len(pos_list)
        if k <= best:          # cannot improve the answer
            continue

        # dynamic programming table:
        # dp[j][d] = length of an arithmetic progression (gap d)
        #            ending at pos_list[j]
        # we store it as an array('H') (unsigned short, 2 bytes)
        dp = [array('H', [0]*(N+1)) for _ in range(k)]

        for i in range(k):
            pi = pos_list[i]
            dpi = dp[i]
            for j in range(i+1, k):
                d = pos_list[j] - pi
                prev = dpi[d]
                if prev == 0:      # no existing sequence, start with length 1
                    prev = 1
                cur = prev + 1     # extend the sequence by pos_list[j]
                if cur > dp[j][d]:
                    dp[j][d] = cur
                    if cur > best:
                        best = cur
                        if best == k:       # cannot get longer within this height
                            break
            if best == k:
                break

    print(best)

if __name__ == "__main__":
    main()