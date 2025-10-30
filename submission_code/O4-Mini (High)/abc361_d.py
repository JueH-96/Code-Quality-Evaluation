import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N = int(input())
    S = input().strip()
    T = input().strip()

    # Precompute bit‐shifts: for physical position i, its 2‐bit field starts at bit 2*i
    shift2 = [2 * i for i in range(N + 2)]

    # Encode a configuration of length N+2 as a 32‐bit integer mask:
    #   00 = empty, 01 = white stone, 10 = black stone
    def encode(line):
        m = 0
        for i, c in enumerate(line):
            if c == 'W':
                m |= (1 << shift2[i])
            else:  # c == 'B'
                m |= (2 << shift2[i])
        # the last two positions (N, N+1) remain 00 (empty)
        return m

    start = encode(S)
    target = encode(T)

    # Quick check
    if start == target:
        print(0)
        return

    # BFS level by level, tracking visited states
    visited = set([start])
    vadd = visited.add
    vcont = visited.__contains__

    curr = [start]
    depth = 0

    while curr:
        depth += 1
        nxt = []
        for mask in curr:
            # locate the empty pair: two adjacent positions p, p+1 with bits == 00
            # we know there is exactly one such adjacent blank pair
            for p in range(N + 1):
                if ((mask >> shift2[p]) & 3) == 0 and ((mask >> shift2[p+1]) & 3) == 0:
                    empty = p
                    break

            # try moving every adjacent‐stones pair (i, i+1) into the empty (empty, empty+1)
            base_clear = mask  # we'll clear the source bits out of this
            for i in range(N + 1):
                # fetch the two stones at i and i+1
                v1 = (mask >> shift2[i]) & 3
                if v1 == 0:
                    continue
                v2 = (mask >> shift2[i+1]) & 3
                if v2 == 0:
                    continue

                # clear them out of their source
                m2 = base_clear & ~((3 << shift2[i]) | (3 << shift2[i+1]))
                # place them into the empty slots
                nm = m2 | (v1 << shift2[empty]) | (v2 << shift2[empty+1])

                if not vcont(nm):
                    if nm == target:
                        print(depth)
                        return
                    vadd(nm)
                    nxt.append(nm)

        curr = nxt

    # unreachable
    print(-1)

# Call main
main()