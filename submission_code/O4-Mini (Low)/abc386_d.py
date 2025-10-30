def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # We will build a single array r (length conceptually N) so that
    # a cell (x,y) is black iff r[x] >= y iff r[y] >= x.
    # From each black C=B at (x,y) we get r[x] >= y and r[y] >= x.
    # Collect these lower‐bounds, enforce symmetry, then check for
    # consistency of the implied "Ferrers diagram" and for whites.
    from collections import defaultdict, deque
    
    LB = defaultdict(int)  # LB[i] = minimal r[i]
    whites = []
    for _ in range(M):
        x,y,c = input().split()
        x = int(x); y = int(y)
        if c == 'B':
            # impose symmetry bounds
            if LB[x] < y:
                LB[x] = y
            if LB[y] < x:
                LB[y] = x
        else:
            whites.append((x,y))
    # Now enforce symmetry closure: whenever LB[x]>=y we must have LB[y]>=x.
    # We'll do a BFS on newly added constraints.
    q = deque(LB.items())
    # We also mark which (x,y) we have already visited in the BFS to avoid loops.
    seen = set(LB.items())
    while q:
        x, rx = q.popleft()
        # for all y up to rx we would need LB[y]>=x.  We cannot loop up to rx
        # if rx is huge.  But if rx > number of distinct keys we already fail,
        # because we would need constraints on unseen indices.
        # Check the critical condition: rx <= count of i with LB[i]>=x,
        # otherwise one of those y in [1..rx] without LB[y]>=x must be missing.
        cnt = 0
        # We count only among those y for which we have a bound LB[y]>=x
        # This is a lower bound on how many y in [1..rx] we could satisfy.
        # If cnt < rx, there's some y in [1..rx] missing, so impossible.
        for y, ry in LB.items():
            if ry >= x:
                cnt += 1
        if cnt < rx:
            print("No")
            return
        # Also enforce per‐point symmetry on the LB dictionary
        # For every y in LB keys with LB[x]>=y, ensure LB[y]>=x
        # We only need to check existing keys, because if y is not a key
        # then LB[y]=0< x would have already failed the count‐check above.
        for y, ry in list(LB.items()):
            if y <= rx and ry < x:
                LB[y] = x
                item = (y, x)
                if item not in seen:
                    seen.add(item)
                    q.append(item)
    
    # Finally we have a symmetric Ferrers‐shape LB that dominates all black constraints.
    # Check all whites: (x,y) must not lie in the black region:
    # i.e. NOT( r[x]>=y AND r[y]>=x ).
    for x,y in whites:
        if LB.get(x,0) >= y and LB.get(y,0) >= x:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()