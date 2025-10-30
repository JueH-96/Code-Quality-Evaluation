def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Compute sumDistinct = sum over subarrays of (number of distinct numbers)
    # Each occurrence A[i] contributes (i - last_occurrence)*(n - i)
    last_occ = {}
    sumDistinct = 0
    for i, a in enumerate(A):
        prev = last_occ.get(a, -1)
        sumDistinct += (i - prev) * (n - i)
        last_occ[a] = i
        
    totSub = n * (n + 1) // 2  # total number of subarrays

    # Build positions dictionary: for each value, store the list of indices (0-indexed)
    positions = {}
    for i, a in enumerate(A):
        positions.setdefault(a, []).append(i)
    
    # Function to compute the number of subarrays that do NOT contain a given value,
    # given its sorted positions list.
    def count_missing(pos_list):
        cnt = 0
        prev = -1
        for p in pos_list:
            gap = p - prev - 1
            cnt += gap * (gap + 1) // 2
            prev = p
        gap = n - prev - 1
        cnt += gap * (gap + 1) // 2
        return cnt

    # Function to compute the number of subarrays that do NOT contain any occurrence
    # of the numbers from the merged sorted positions list.
    def count_missing_merged(list1, list2):
        merged = []
        i = j = 0
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                merged.append(list1[i])
                i += 1
            else:
                merged.append(list2[j])
                j += 1
        while i < len(list1):
            merged.append(list1[i])
            i += 1
        while j < len(list2):
            merged.append(list2[j])
            j += 1
        cnt = 0
        prev = -1
        for p in merged:
            gap = p - prev - 1
            cnt += gap * (gap + 1) // 2
            prev = p
        gap = n - prev - 1
        cnt += gap * (gap + 1) // 2
        return cnt

    # Now, sum up over all consecutive integer pairs (x, x+1)
    totalConsecutive = 0
    # We consider x from 1 to n-1; note that A's values are in [1, n]
    for x in range(1, n):
        if (x in positions) and ((x+1) in positions):
            F_x = count_missing(positions[x])
            F_y = count_missing(positions[x+1])
            F_xy = count_missing_merged(positions[x], positions[x+1])
            # Count of subarrays that contain both x and x+1 (by inclusion-exclusion)
            cntBoth = totSub - F_x - F_y + F_xy
            totalConsecutive += cntBoth

    # Our answer is sumDistinct (the sum of |S| over subarrays)
    # minus totalConsecutive (the sum over subarrays of the count of consecutive pairs in S)
    ans = sumDistinct - totalConsecutive
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()