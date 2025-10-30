def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        A = data[index]
        index += 1
        B = data[index]
        index += 1

        # Build lists of indices (1-indexed) for pieces in A and required squares in B.
        arrA = [i + 1 for i, ch in enumerate(A) if ch == '1']
        arrB = [i + 1 for i, ch in enumerate(B) if ch == '1']

        # If there are not enough pieces to cover B's ones, it is impossible.
        if len(arrA) < len(arrB):
            results.append("-1")
            continue

        # The key observation:
        # In one operation, each piece (if not at the chosen pivot) moves one step closer to
        # the pivot. Hence, moving a piece from position a to b requires exactly |a - b| operations.
        # We would like to cover each target square (in B with value '1')
        # with at least one piece from A.
        # Since the operations move pieces simultaneously and extra pieces can be merged into
        # target squares without extra penalty, the minimum number of moves needed is
        # at least the maximum shift needed for some matching between a subset of pieces in A and all indices in B.
        #
        # In a sorted order matching (i.e. pairing the k-th piece used from A with the k-th target in B),
        # the minimal required maximum moves is the minimum m such that we can find an increasing subsequence
        # of indices in arrA where for each target b in arrB the chosen a satisfies |a - b| ≤ m.
        #
        # We can decide if a candidate m works with a greedy matching:
        def can(m):
            j = 0  # pointer for arrA
            for b in arrB:
                # We need a piece from A that is at least b - m.
                while j < len(arrA) and arrA[j] < b - m:
                    j += 1
                if j >= len(arrA):
                    return False
                # If the found piece is too far from b (i.e. arrA[j] > b + m) then m is not enough.
                if abs(arrA[j] - b) > m:
                    return False
                j += 1  # assign this piece to b
            return True

        # Binary search for the minimum m from 0 up to n (the maximum absolute difference
        # is at most n).
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if can(mid):
                hi = mid
            else:
                lo = mid + 1
        
        results.append(str(lo))
    
    sys.stdout.write("
".join(results))
    
if __name__ == "__main__":
    main()