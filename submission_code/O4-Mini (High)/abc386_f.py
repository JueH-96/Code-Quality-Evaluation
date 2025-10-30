import sys
from collections import deque

def main():
    data = sys.stdin
    line = data.readline()
    if not line:
        return
    K = int(line.strip())
    S = data.readline().rstrip('
')
    T = data.readline().rstrip('
')
    n = len(S)
    m = len(T)
    # Quick equal check
    if S == T:
        print("Yes")
        return
    # Trim common prefix
    p = 0
    limit = n if n < m else m
    while p < limit and S[p] == T[p]:
        p += 1
    if p > 0:
        S = S[p:]
        T = T[p:]
        n -= p
        m -= p
    # If one is empty now
    if n == 0 or m == 0:
        # Only insert/delete needed
        if n + m <= K:
            print("Yes")
        else:
            print("No")
        return
    # Trim common suffix
    suf = 0
    # note: after prefix trim, n,m updated
    while suf < n and suf < m and S[n-1-suf] == T[m-1-suf]:
        suf += 1
    if suf > 0:
        S = S[:n-suf]
        T = T[:m-suf]
        n -= suf
        m -= suf
    # If one is empty after suffix trim
    if n == 0 or m == 0:
        if n + m <= K:
            print("Yes")
        else:
            print("No")
        return
    # If length difference too big
    if abs(n - m) > K:
        print("No")
        return

    # BFS with greedy extension
    dq = deque()
    visited = set()
    # initial greedy match
    i0 = 0
    j0 = 0
    while i0 < n and j0 < m and S[i0] == T[j0]:
        i0 += 1
        j0 += 1
    if i0 >= n and j0 >= m:
        print("Yes")
        return
    dq.append((i0, j0, 0))
    visited.add((i0, j0))

    s = S
    t = T
    max_k = K
    while dq:
        i, j, d = dq.popleft()
        if d >= max_k:
            continue
        d2 = d + 1
        # Deletion (drop one char from s)
        if i < n:
            ii = i + 1
            jj = j
            while ii < n and jj < m and s[ii] == t[jj]:
                ii += 1; jj += 1
            if ii >= n and jj >= m:
                print("Yes")
                return
            st = (ii, jj)
            if st not in visited:
                visited.add(st)
                dq.append((ii, jj, d2))
        # Insertion (insert one char into s)
        if j < m:
            ii = i
            jj = j + 1
            while ii < n and jj < m and s[ii] == t[jj]:
                ii += 1; jj += 1
            if ii >= n and jj >= m:
                print("Yes")
                return
            st = (ii, jj)
            if st not in visited:
                visited.add(st)
                dq.append((ii, jj, d2))
        # Substitution
        if i < n and j < m and s[i] != t[j]:
            ii = i + 1
            jj = j + 1
            while ii < n and jj < m and s[ii] == t[jj]:
                ii += 1; jj += 1
            if ii >= n and jj >= m:
                print("Yes")
                return
            st = (ii, jj)
            if st not in visited:
                visited.add(st)
                dq.append((ii, jj, d2))

    # No solution within K
    print("No")

if __name__ == "__main__":
    main()