import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    from collections import Counter
    ca = Counter()
    cb = Counter()
    duA = duB = 0
    maxA = 0
    maxB = 0
    for x in A:
        if x == -1:
            duA += 1
        else:
            ca[x] += 1
            if x > maxA: maxA = x
    for x in B:
        if x == -1:
            duB += 1
        else:
            cb[x] += 1
            if x > maxB: maxB = x
    na = sum(ca.values())
    nb = sum(cb.values())
    # Number of known values that must be covered by unknowns
    need = max(na - duB, nb - duA)
    # If we can cover all knowns by unknowns, we can choose S arbitrarily ≥ maxA,maxB
    if need <= 0:
        print("Yes")
        return
    # S must be >= max(maxA, maxB) for non-negative assignments
    MV = maxA if maxA > maxB else maxB
    # Try to find S = x+y with enough known-known matches
    m = {}
    # Iterate over distinct known A-values and known B-values
    # ca[x] and cb[y] are their frequencies
    for x, cx in ca.items():
        # For this x, early skip if even pairing with all B cannot reach need
        # But skip such prune: not clear, skip
        for y, cy in cb.items():
            S = x + y
            if S < MV:
                continue
            # Each value-class match contributes min(cx, cy) towards matched known-known
            # only for this pair of value-classes
            d = cx if cx < cy else cy
            prev = m.get(S)
            if prev is None:
                cnt = d
            else:
                cnt = prev + d
            # If we reach required matches, it's feasible
            if cnt >= need:
                print("Yes")
                return
            m[S] = cnt
    # No S reached needed matches
    print("No")

if __name__ == "__main__":
    main()