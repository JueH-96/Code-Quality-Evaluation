import sys
from array import array
from collections import deque

def main():
    input = sys.stdin.readline
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Double the array for circular handling
    twoN = 2 * N
    B = A + A
    total = sum(A)
    # Number of bits needed for binary‐lifting up to K
    maxbit = K.bit_length()
    
    def feasible(mid):
        """
        Return True iff we can cut the circular cake into K segments
        each of mass >= mid.
        """
        # Build nxt[1..2N], nxt[i] = smallest j>i with sum(B[i..j-1])>=mid
        nxt = [0] * (twoN + 2)
        cur = 0
        j = 1
        for i in range(1, twoN + 1):
            while j <= twoN and cur < mid:
                cur += B[j-1]
                j += 1
            nxt[i] = j if cur >= mid else (twoN + 1)
            cur -= B[i-1]
        nxt[twoN+1] = twoN+1
        
        # Build binary‐lifting table dp[b][i]
        dp = [array('I', nxt)]
        for b in range(1, maxbit):
            prev = dp[b-1]
            curr = array('I', [0]) * (twoN + 2)
            curr[twoN+1] = twoN+1
            for i in range(1, twoN + 1):
                p = prev[i]
                curr[i] = prev[p] if p <= twoN+1 else (twoN+1)
            dp.append(curr)
        
        # Try each start i=1..N, see if K jumps fit in window of length N
        for i in range(1, N+1):
            pos = i
            limit = i + N
            bit = 0
            rem = K
            while rem:
                if rem & 1:
                    pos = dp[bit][pos]
                    if pos > limit:
                        break
                rem >>= 1
                bit += 1
            else:
                # never broke => p_K(i) <= i+N
                return True
        return False
    
    # Binary search for the maximum feasible minimum mass x
    lo, hi = 0, total + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    x = lo
    
    # Now re‐build nxt and dp for mid = x, so that we can gather all valid starts
    # and mark every cut line they ever use.
    nxt = [0] * (twoN + 2)
    cur = 0
    j = 1
    for i in range(1, twoN + 1):
        while j <= twoN and cur < x:
            cur += B[j-1]
            j += 1
        nxt[i] = j if cur >= x else (twoN + 1)
        cur -= B[i-1]
    nxt[twoN+1] = twoN+1
    
    dp = [array('I', nxt)]
    for b in range(1, maxbit):
        prev = dp[b-1]
        curr = array('I', [0]) * (twoN + 2)
        curr[twoN+1] = twoN+1
        for i in range(1, twoN + 1):
            p = prev[i]
            curr[i] = prev[p] if p <= twoN+1 else (twoN+1)
        dp.append(curr)
    
    # Collect all valid starts S = { i : p_K(i) <= i+N }
    S = []
    for i in range(1, N+1):
        pos = i
        limit = i + N
        bit = 0
        rem = K
        while rem:
            if rem & 1:
                pos = dp[bit][pos]
                if pos > limit:
                    break
            rem >>= 1
            bit += 1
        else:
            S.append(i)
    
    # BFS from all starts in S (depth <= K-1) along the nxt‐edges
    # to find all j = p_t(i) for t<K and i∈S
    dist = [-1] * (twoN + 2)
    q = deque()
    for s in S:
        dist[s] = 0
        q.append(s)
    maxd = K - 1
    while q:
        u = q.popleft()
        d = dist[u]
        if d == maxd:
            continue
        v = nxt[u]
        if v <= twoN and dist[v] < 0:
            dist[v] = d + 1
            q.append(v)
    
    # Mark which original cut‐lines 1..N ever appear
    used = [False] * (N + 1)
    for j in range(1, twoN + 1):
        if dist[j] >= 0:
            # segment boundary at j means cut‐line = (j-1) mod N,
            # where 0->N.
            c = j - 1
            if c > N:
                c %= N
            if c == 0:
                c = N
            used[c] = True
    
    # Count how many lines are never used
    never_cut = sum(1 for i in range(1, N+1) if not used[i])
    
    print(x, never_cut)

if __name__ == "__main__":
    main()