import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    S = list(input().strip())
    T = list(input().strip())

    # We'll process T in order, but we'll maintain the best possible final string
    # by a two‐pointer greedy that tries to improve S from left to right
    # while respecting the order of T and allowing us to "waste" small T's on
    # the rightmost end when they can't help.
    #
    # Concretely, we keep a deque for the future operations.  Whenever we're
    # at position i in S, we look ahead in T (at our current pointer) for the
    # largest digit we *can* still use at or after that pointer that would
    # strictly exceed S[i].  If we find one, we assign it.  Otherwise we keep
    # S[i] as is.  Any T's we skip in pursuit of a better digit for S[i] we
    # must "waste" somewhere; the optimal place to waste them is always at
    # position N (the least significant place), because that does least harm.
    #
    # This runs in O(N + M) time by maintaining for each digit 1..9 a queue
    # of its positions in T, and a Fenwick tree (BIT) to count how many T's
    # we've skipped before a given index.

    from collections import deque

    # Build queues of positions for each digit in T
    pos = {str(d): deque() for d in range(1,10)}
    for idx, ch in enumerate(T):
        pos[ch].append(idx)

    # Fenwick tree to count "used or skipped" T's in prefixes
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0]*(n+1)
        def add(self, i, v):
            i += 1
            while i <= self.n:
                self.fw[i] += v
                i += i & -i
        def sum(self, i):
            # sum [0..i)
            s = 0
            while i > 0:
                s += self.fw[i]
                i -= i & -i
            return s
        def range_sum(self, l, r):
            return self.sum(r) - self.sum(l)

    BIT = Fenwick(M)
    t_ptr = 0  # how many T's we've actually committed (used or wasted)

    S = list(S)
    for i in range(N):
        best_d = None
        best_idx = None
        # look for the largest digit > S[i] that we can still reach in T
        for d in range(9, int(S[i]), -1):
            dch = str(d)
            if not pos[dch]:
                continue
            j = pos[dch][0]  # earliest occurrence in T
            # how many T's we've already passed before j?
            used_before = BIT.sum(j+1)
            # the actual operation index if we tried to use T[j] now would be
            op_idx = t_ptr + (j+1 - used_before)
            # we just need that op_idx <= M, which is automatic as we never overshoot
            best_d = dch
            best_idx = j
            break

        if best_d is not None:
            # waste all smaller/skipped Ts up to best_idx
            while t_ptr < M:
                # the very next T we haven't accounted for is at real index
                # let's find the smallest real index r so that we've not BIT'ed it yet
                # but simpler: if the next un-used/skipped is best_idx, stop
                # else waste it.
                # We'll do a binary search on BIT to locate the next free slot
                lo, hi = 0, M
                while lo < hi:
                    mid = (lo + hi)//2
                    if mid+1 - BIT.sum(mid+1) >= 1:
                        hi = mid
                    else:
                        lo = mid+1
                r = lo
                if r == best_idx:
                    break
                # waste T[r]
                BIT.add(r, 1)
                t_ptr += 1
            # now actually use T[best_idx]
            BIT.add(best_idx, 1)
            t_ptr += 1
            pos[best_d].popleft()
            S[i] = best_d
        # else we leave S[i] as is

    # any leftover Ts we must waste (they go to pos N, but that doesn't change S)
    # so we don't need to do anything more

    print(''.join(S))


if __name__ == "__main__":
    main()