import sys

# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------
def build_next(N, pref, need):
    """
    For every boundary (line) position i (0 <= i < 2N) return the smallest
    index j (i < j <= 2N) such that the total mass between i and j is at least
    `need` and j-i <= N (i.e. no more than one full round).  
    If such j does not exist –1 is stored instead.
    """
    size = 2 * N
    nxt = [-1] * size
    r = 0                       # right pointer (line index)
    for i in range(size):
        if r <= i:
            r = i + 1
        # advance r until segment [i, r) is heavy enough
        while r <= size and pref[r] - pref[i] < need:
            r += 1
        # store answer if it is inside one round and inside the duplicated area
        if r <= size and r - i <= N and r < size:
            nxt[i] = r
    return nxt


def build_jumps(nxt, max_bits):
    """
    Build doubling table jumps[p][i] – 2**p applications of nxt starting
    from i.  jumps[0] is simply nxt itself.
    """
    jumps = [nxt]
    for _ in range(1, max_bits):
        prev = jumps[-1]
        size = len(prev)
        cur = [-1] * size
        for i in range(size):
            j = prev[i]
            cur[i] = -1 if j == -1 else prev[j]
        jumps.append(cur)
    return jumps


def reach(i, K, jumps):
    """Return position after K steps from i using pre-built doubling table."""
    bit = 0
    idx = i
    k = K
    while k and idx != -1:
        if k & 1:
            idx = jumps[bit][idx]
        k >>= 1
        bit += 1
    return idx


def possible(N, K, pref, need, max_bits):
    """
    True if there exists a starting cut line that allows division where every
    share is at least `need`.
    """
    nxt = build_next(N, pref, need)
    jumps = build_jumps(nxt, max_bits)

    for s in range(N):                      # only real (first) N lines matter
        last = reach(s, K, jumps)
        if last != -1 and last - s <= N:    # finished within one full circle
            return True
    return False
# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    S = sum(A)

    size = 2 * N
    pref = [0] * (size + 1)
    for i in range(size):
        pref[i + 1] = pref[i] + A[i % N]

    max_bits = K.bit_length()

    # --- binary search for the maximum minimal share ------------------------
    low, high = 0, S // K
    while low < high:
        mid = (low + high + 1) // 2
        if possible(N, K, pref, mid, max_bits):
            low = mid
        else:
            high = mid - 1
    best = low                           # x  in the statement

    # --- build structures for the optimal value -----------------------------
    nxt_opt = build_next(N, pref, best)
    jumps_opt = build_jumps(nxt_opt, max_bits)

    feasible_starts = 0
    for s in range(N):
        last = reach(s, K, jumps_opt)
        if last != -1 and last - s <= N:
            feasible_starts += 1

    never_cut = N - feasible_starts      # y  in the statement
    print(best, never_cut)


if __name__ == "__main__":
    main()