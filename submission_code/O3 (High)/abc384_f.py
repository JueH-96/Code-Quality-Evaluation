import sys
from collections import defaultdict

# ------------------------------------------------------------
def one_S_k(A, W):
    """
    compute S_k   (definition (3))   for W = 2^k
    """
    cnt = defaultdict(int)
    sm  = defaultdict(int)
    mask = W - 1
    # build residue classes
    for x in A:
        r = x & mask          # x mod W  (W is power of two)
        cnt[r] += 1
        sm[r]  += x

    total = 0
    visited = set()
    for r in cnt:
        if r in visited:
            continue
        s = (-r) & mask       # partner residue (W - r) mod W
        if s == r:            # only r = 0 or W/2
            total += (cnt[r] + 1) * sm[r]
            visited.add(r)
        elif s in cnt:
            total += cnt[r] * sm[s] + cnt[s] * sm[r]
            visited.add(r)
            visited.add(s)
        else:                 # partner residue not present
            visited.add(r)
    return total // W         # integer division – W divides total


# ------------------------------------------------------------
def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    # maximum k so that 2^k ≤ max possible sum (2*10^7)
    MAX_SUM = 2 * 10_000_000
    k_max = 0
    while (1 << k_max) <= MAX_SUM:
        k_max += 1
    k_max -= 1            # last k with 2^k ≤ MAX_SUM

    S = [0] * (k_max + 2)       # also room for k_max+1 (will be 0)
    W = 1
    for k in range(k_max + 1):
        S[k] = one_S_k(A, W)
        W <<= 1                 # next power of two

    # S[k_max+1] already 0

    ans = 0
    for k in range(k_max + 1):
        ans += S[k] - 2 * S[k + 1]

    print(ans)


# ------------------------------------------------------------
if __name__ == "__main__":
    main()