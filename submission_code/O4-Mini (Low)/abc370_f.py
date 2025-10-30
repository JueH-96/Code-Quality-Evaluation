import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline
    N,K = map(int,input().split())
    A = list(map(int,input().split()))
    # build doubled prefix sums
    S = [0]*(2*N+1)
    for i in range(2*N):
        S[i+1] = S[i] + A[i%N]
    # binary search on x
    def can(x):
        # compute next[i]: minimal j>i with sum[i:j]>=x
        nxt = [2*N+1]*(2*N)
        j = 0
        for i in range(2*N):
            if j < i: j = i
            while j < 2*N and S[j+1]-S[i] < x:
                j += 1
            if j<2*N and S[j+1]-S[i]>=x:
                nxt[i] = j+1
        # build binary lifting
        LOG = K.bit_length()
        up = [nxt[:] ]
        for b in range(1,LOG):
            prev = up[b-1]
            cur = [2*N+1]*(2*N)
            for i in range(2*N):
                ni = prev[i]
                cur[i] = prev[ni] if ni<2*N else 2*N+1
            up.append(cur)
        # check any start in [0,N) works
        for i in range(N):
            pos = i
            rem = K
            for b in range(LOG):
                if rem&1:
                    pos = up[b][pos] if pos<2*N else 2*N+1
                rem >>=1
                if pos>i+N:
                    break
            else:
                # succeeded
                return True
        return False

    lo,hi = 0,sum(A)+1
    while lo+1<hi:
        mid = (lo+hi)//2
        if can(mid):
            lo = mid
        else:
            hi = mid
    x = lo

    # Now compute y: number of never-cut cut-lines.
    # A cut-line i (between i and i+1) is cut if some optimal
    # segmentation places a boundary there. We find all boundaries
    # that can occur in some segmentation achieving min-sum x.
    # We mark all possible boundary positions.
    # We'll reuse next[] for x.
    # First build next and lifting again:
    S = S  # same
    nxt = [2*N+1]*(2*N)
    j = 0
    for i in range(2*N):
        if j < i: j = i
        while j < 2*N and S[j+1]-S[i] < x:
            j += 1
        if j<2*N and S[j+1]-S[i]>=x:
            nxt[i] = j+1
    LOG = K.bit_length()
    up = [nxt[:]]
    for b in range(1,LOG):
        prev = up[b-1]
        cur = [2*N+1]*(2*N)
        for i in range(2*N):
            ni = prev[i]
            cur[i] = prev[ni] if ni<2*N else 2*N+1
        up.append(cur)
    # determine which start positions are valid
    valid_start = [False]*N
    for i in range(N):
        pos = i
        rem = K
        ok = True
        for b in range(LOG):
            if rem&1:
                pos = up[b][pos]
            rem >>=1
            if pos>i+N:
                ok = False
                break
        if ok:
            valid_start[i] = True

    # now for each valid start i, we trace the jumps once
    # but K can be large; however total tracing across all valid starts is O(N log N)
    # we will simulate step by step but reuse binary lifts to jump bits,
    # and collect all intermediate boundaries.
    isCut = [False]*N
    from collections import deque
    # For each valid start, do a BFS in jump-power space to mark all reached nodes <=i+N
    # We'll do it by DFS on bits of K, marking edges. Actually we want all boundary positions where a jump occurs.
    # We'll for each start simulate greedily K jumps by bits, but also record each individual jump.
    # That is O(N log K). Acceptable.
    for i in range(N):
        if not valid_start[i]: continue
        # we reconstruct the path of K jumps bit by bit
        def go(start,rem,base):
            # recursively process rem jumps from start
            if rem==0: return
            b = (rem.bit_length()-1)
            if rem == (1<<b):
                # a single block of size 2^b: but we need record all intermediate smaller jumps?
                # we must go down to single jumps
                if rem==1:
                    # single jump: start->nxt[start]
                    to = nxt[start]
                    if to<=base+N:
                        cutpos = (to-1)%N  # boundary after piece to
                        isCut[cutpos] = True
                    return
                # split into two halves
                half = rem//2
                go(start,half,base)
                # find mid position
                pos = start
                r = half
                for bb in range(LOG):
                    if (r>>bb)&1:
                        pos = up[bb][pos]
                go(pos,half,base)
            else:
                # remove highest bit
                highest = 1<<b
                go(start,highest,base)
                pos = start
                r = highest
                for bb in range(LOG):
                    if (r>>bb)&1:
                        pos = up[bb][pos]
                go(pos, rem-highest,base)
        go(i,K,i)
    # cut-lines that never get cut:
    y = sum(1 for i in range(N) if not isCut[i])
    print(x,y)

if __name__ == "__main__":
    main()