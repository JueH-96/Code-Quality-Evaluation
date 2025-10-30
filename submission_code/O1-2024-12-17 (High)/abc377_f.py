def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Read piece positions
    rowSet = set()
    colSet = set()
    diagSet = set()
    antiSet = set()
    
    idx = 2
    for _ in range(M):
        a = int(input_data[idx]); b = int(input_data[idx+1])
        idx += 2
        rowSet.add(a)
        colSet.add(b)
        diagSet.add(a+b)
        antiSet.add(a-b)
        
    # Convert to sorted lists
    rowList = sorted(rowSet)
    colList = sorted(colSet)
    diagList = sorted(diagSet)
    antiList = sorted(antiSet)
    
    # Precompute sizes
    rCount = len(rowList)
    cCount = len(colList)
    # Number of squares that survive row/column blocking
    S = (N - rCount) * (N - cCount)
    
    # For fast membership checks when computing intersections
    rowSetLookup = set(rowList)
    colSetLookup = set(colList)
    
    # ----------------------------------------------------------------------
    # Helper: count how many unique values from two sorted lists fall in [L..R].
    # Both lists may contain duplicates internally; we only count distinct values.
    def count_unique_in_range(sortedA, sortedB, L, R):
        # Two-pointer merge of sortedA and sortedB (both ascending).
        i = 0
        j = 0
        lenA = len(sortedA)
        lenB = len(sortedB)
        last_val = None
        count_in_interval = 0
        
        while i < lenA or j < lenB:
            if i < lenA:
                valA = sortedA[i]
            else:
                valA = 10**15  # something bigger than any feasible index
            
            if j < lenB:
                valB = sortedB[j]
            else:
                valB = 10**15
            
            if valA < valB:
                this_val = valA
                i += 1
            elif valB < valA:
                this_val = valB
                j += 1
            else:
                # valA == valB
                this_val = valA
                i += 1
                j += 1
            
            # Check distinct and range
            if this_val != last_val:
                if L <= this_val <= R:
                    count_in_interval += 1
                last_val = this_val
        
        return count_in_interval
    
    # ----------------------------------------------------------------------
    # Count how many squares in pI x pJ (unused rows & cols)
    # land on the diagonals of the existing pieces.
    # i + j = d
    # We'll exclude any i or j that appear in rowSet or colSet.
    # S_D = sum over each diag d of how many (i,j) in pI x pJ satisfy i+j=d
    #
    # We do that by: i = [max(1,d-N)..min(N,d-1)], j = d - i
    # i must not be in rowSet, j must not be in colSet.
    # We can transform "i not in rowSet and d-i not in colSet" into
    # "i not in (rowSet union {d - c for c in colSet})".
    
    def count_unblocked_diag(d):
        L = max(1, d - N)
        R = min(N, d - 1)
        if L > R:
            return 0
        
        # Build offsetAsc for the columns: offsetAsc = sorted(d - col for col in colList)
        # colList is ascending, so (d - colList) produces descending => then reverse
        offsetDesc = [d - c for c in colList]  # descending if colList is ascending
        offsetDesc.reverse()                  # now it's ascending
        # Merge rowList and offsetAsc, count how many unique fall in [L..R]
        blocked_count = count_unique_in_range(rowList, offsetDesc, L, R)
        total_in_range = (R - L + 1)
        return total_in_range - blocked_count
    
    # Sum up all diagonals
    S_D = 0
    for d in diagList:
        S_D += count_unblocked_diag(d)
    
    # ----------------------------------------------------------------------
    # Similarly for anti diagonals i - j = c => j = i - c
    # i in [1..N], j in [1..N], so i in [max(1, c+1) .. min(N, c+N)].
    # We exclude i in rowSet or j in colSet => i not in rowSet and i-c not in colSet.
    # That means i not in (rowSet union {c + col for col in colSet}).
    
    def count_unblocked_anti(c):
        L = max(1, c + 1)
        R = min(N, c + N)
        if L > R:
            return 0
        
        # offsetAsc = [c + col for col in colList], colList ascending => sum is ascending
        offsetAsc = [c + col for col in colList]
        blocked_count = count_unique_in_range(rowList, offsetAsc, L, R)
        total_in_range = (R - L + 1)
        return total_in_range - blocked_count
    
    # Sum up all anti diagonals
    S_A = 0
    for c in antiList:
        S_A += count_unblocked_anti(c)
    
    # ----------------------------------------------------------------------
    # Now the intersection S_D ∩ S_A: squares in pI x pJ that lie on both
    # a diag d in diagSet and an anti c in antiSet. That means:
    #   i + j = d  and  i - j = c
    # => Adding => 2i = d + c => i = (d + c)//2
    #    Subtract => 2j = d - c => j = (d - c)//2
    # We must have (d+c) even, i in [1..N], j in [1..N],
    # and also i not in rowSet, j not in colSet.
    
    S_DA = 0
    # For speed, convert rowSetLookup, colSetLookup to sets (already done).
    for d in diagList:
        for c in antiList:
            # Check parity
            if (d + c) & 1:
                # If d+c is odd => no integer i
                continue
            i = (d + c) >> 1
            j = (d - c) >> 1
            if 1 <= i <= N and 1 <= j <= N:
                if i not in rowSetLookup and j not in colSetLookup:
                    S_DA += 1
    
    # Finally, the number of squares in S that also lie on
    # at least one existing diag or anti is S_D + S_A - S_DA
    # so the number of squares in S that do NOT lie on any diag or anti is:
    #   S - (S_D + S_A - S_DA)
    
    ans = S - (S_D + S_A - S_DA)
    
    print(ans)

# Do not forget to call main()!
main()