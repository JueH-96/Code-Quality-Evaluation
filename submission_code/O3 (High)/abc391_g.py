#!/usr/bin/env python3
MOD = 998244353

# ---------------------------------------------
# pack / unpack of an array of length <= 10
# 4 bits per component ( fits 0 … 10 )
# ---------------------------------------------
def pack(arr):                 # arr length = n ≤ 10
    code = 0
    for x in arr[::-1]:
        code = (code << 4) | x
    return code

def unpack(code, n):
    arr = [0]*n
    for i in range(n):
        arr[i] = code & 0xf
        code >>= 4
    return arr[::-1]           # back to left→right


def main() -> None:
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    S    = sys.stdin.readline().strip()
    n = N

    # ----------------  distinct letters ----------------
    special = sorted(set(S))
    L = len(special) + 1                       # + OTHER
    OTHER_ID = L - 1                           # last index is OTHER

    # map letter → id  (0 … len(special)-1) else OTHER_ID
    letter_id = {c:i for i,c in enumerate(special)}

    multiplicity = [1]*L
    multiplicity[OTHER_ID] = 26 - len(special)

    # --------------  enumerate all reachable states -------------
    from collections import defaultdict, deque

    init_state = pack([0]*(n+1))
    states = [init_state]                      # list keeps order
    index  = {init_state:0}                    # state → idx
    trans  = []                                # trans[idx][letter] = idx2

    q = deque([init_state])
    while q:
        st = q.popleft()
        vec = unpack(st, n+1)
        row = [None]*L

        # pre-compute once, store
        for lid in range(L):
            if lid == OTHER_ID:
                c = None                       # “letter not in S”
            else:
                c = special[lid]

            new = [0]*(n+1)
            for i in range(1, n+1):
                if c is not None and S[i-1] == c:
                    new[i] = vec[i-1] + 1
                else:
                    new[i] = max(vec[i], new[i-1])

            code = pack(new)
            if code not in index:
                index[code] = len(states)
                states.append(code)
                q.append(code)
            row[lid] = index[code]
        trans.append(row)

    S_cnt = len(states)

    # ---------------- DP over M positions ----------------
    cur = [0]*S_cnt
    cur[0] = 1

    for _ in range(M):
        nxt = [0]*S_cnt
        for s_idx, ways in enumerate(cur):
            if ways == 0:
                continue
            tr = trans[s_idx]
            for lid in range(L):
                nxt[tr[lid]] = (nxt[tr[lid]] + ways * multiplicity[lid]) % MOD
        cur = nxt

    # ---------------- collect answers -------------------
    ans = [0]*(n+1)
    for s_idx, ways in enumerate(cur):
        if ways == 0:
            continue
        vecN = states[s_idx] & 0xf            # last 4 bits  = d[N]
        ans[vecN] = (ans[vecN] + ways) % MOD

    print(*ans)

if __name__ == '__main__':
    main()