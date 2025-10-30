#!/usr/bin/env python3
import sys
from array import array

def main():
    data = sys.stdin
    line = data.readline().strip()
    if not line:
        return
    N = int(line)
    grid = [data.readline().rstrip('
') for _ in range(N)]
    # Flatten grid and find player positions
    K = N * N
    empty = [True] * K
    players = []
    for i in range(N):
        row = grid[i]
        base = i * N
        for j in range(N):
            ch = row[j]
            idx = base + j
            if ch == '#':
                empty[idx] = False
            elif ch == 'P':
                players.append(idx)
    # Two players
    p1, p2 = players

    # Precompute moves for each direction
    fU = [0]*K; fD = [0]*K; fL = [0]*K; fR = [0]*K
    for pos in range(K):
        if empty[pos]:
            i = pos // N
            j = pos - i*N
            # up
            if i > 0 and empty[pos - N]:
                fU[pos] = pos - N
            else:
                fU[pos] = pos
            # down
            if i < N-1 and empty[pos + N]:
                fD[pos] = pos + N
            else:
                fD[pos] = pos
            # left
            if j > 0 and empty[pos - 1]:
                fL[pos] = pos - 1
            else:
                fL[pos] = pos
            # right
            if j < N-1 and empty[pos + 1]:
                fR[pos] = pos + 1
            else:
                fR[pos] = pos
        else:
            # obstacle maps to itself (won't be used for players)
            fU[pos] = pos
            fD[pos] = pos
            fL[pos] = pos
            fR[pos] = pos

    # Precompute prefix sums for triangular indexing
    # For a <= b, idx = sum_{t=0..a-1}(K-t) + (b-a)
    #        = a*K - a*(a-1)//2 + (b-a)
    prefix = [0]*K
    for a in range(K):
        prefix[a] = a*K - (a*(a-1)//2)

    # Size of triangular visited array
    Vsize = (K*(K+1))//2
    visited = bytearray(Vsize)

    # Bits to pack two positions into one int
    bits = (K-1).bit_length()
    mask = (1 << bits) - 1

    # Initial state (order positions so a<=b)
    if p1 <= p2:
        a0, b0 = p1, p2
    else:
        a0, b0 = p2, p1
    idx0 = prefix[a0] + (b0 - a0)
    visited[idx0] = 1

    # Initialize frontier
    arr_type = array
    cur = arr_type('I', [ (a0 << bits) | b0 ])
    depth = 0

    # Local references for speed
    fU_ = fU; fD_ = fD; fL_ = fL; fR_ = fR
    prefix_ = prefix; visited_ = visited
    bits_ = bits; mask_ = mask

    # BFS
    while cur:
        # build next frontier
        nxt = arr_type('I')
        for v in cur:
            p1 = v >> bits_
            p2 = v & mask_
            # check meeting
            if p1 == p2:
                print(depth)
                return
            # four directions
            # up
            np1 = fU_[p1]; np2 = fU_[p2]
            if np1 > np2:
                t = np1; np1 = np2; np2 = t
            idx = prefix_[np1] + (np2 - np1)
            if not visited_[idx]:
                visited_[idx] = 1
                nxt.append((np1 << bits_) | np2)
            # down
            np1 = fD_[p1]; np2 = fD_[p2]
            if np1 > np2:
                t = np1; np1 = np2; np2 = t
            idx = prefix_[np1] + (np2 - np1)
            if not visited_[idx]:
                visited_[idx] = 1
                nxt.append((np1 << bits_) | np2)
            # left
            np1 = fL_[p1]; np2 = fL_[p2]
            if np1 > np2:
                t = np1; np1 = np2; np2 = t
            idx = prefix_[np1] + (np2 - np1)
            if not visited_[idx]:
                visited_[idx] = 1
                nxt.append((np1 << bits_) | np2)
            # right
            np1 = fR_[p1]; np2 = fR_[p2]
            if np1 > np2:
                t = np1; np1 = np2; np2 = t
            idx = prefix_[np1] + (np2 - np1)
            if not visited_[idx]:
                visited_[idx] = 1
                nxt.append((np1 << bits_) | np2)
        depth += 1
        cur = nxt

    # If we exhaust BFS without meeting
    print(-1)

if __name__ == "__main__":
    main()