def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Explanation:
    #
    # We are given a sequence A of length n. For a subarray A[L:R],
    # let D be the distinct numbers that appear. An "operation" (erase step)
    # can remove a set of numbers if it is a consecutive block of integers.
    # Since you can only remove a consecutive block (like [l, r]), the best strategy is
    # to remove as many consecutive numbers (in the natural order) as possible.
    # Thus the minimal number of operations is exactly the number of blocks in the sorted distinct set.
    #
    # Notice that if D has k numbers and there are c pairs of adjacent numbers in D that 
    # are consecutive (i.e. d_i+1 = d_{i+1}), then the number of blocks is k-c.
    # Equivalently: f(L,R) = (# distinct numbers in A[L:R]) - (# pairs of consecutive integers appearing).
    # 
    # Hence, the final answer we need is:
    #   Sum_{subarray} f(L,R)
    # = Sum_{subarray} [ (# distinct numbers) ] - Sum_{subarray}[ (# consecutive pairs among distinct numbers) ].
    #
    # We can sum over all subarrays by linearity:
    # (1) Sum_{subarray} (# distinct numbers) = Sum_{v=1}^{n} (number of subarrays in which v appears)
    # (2) For each candidate consecutive pair (v, v+1), 
    #     the number of subarrays that contain both v and v+1 is exactly the extra "saving"
    #     (they join into one block) which reduces the cost.
    #
    # So answer = (sum_{v} count(v)) - (sum_{v=1}^{n-1} count(v and v+1)),
    # where count(v) is the number of subarrays (of the whole array A) that contain value v,
    # and count(v and v+1) is for the pair of consecutive values.
    #
    # How to count "subarrays that contain a given set S"?
    #
    # A standard trick is to use the “gap method.” For a sorted list of positions lst where a given number appears,
    # add two sentinels: 0 (before the array) and n+1 (after the array). Then the subarrays that do NOT include the number
    # correspond to subarrays entirely in the gaps between these positions. For a gap of length L, there are L*(L+1)//2 subarrays.
    # Since the total number of subarrays is T = n*(n+1)//2, the number of subarrays that contain the number is T minus
    # the sum over all gaps.
    #
    # We define:
    #   count_subarrays(lst) = T - sum( gap*(gap+1)//2 over all gaps )
    # where the gaps are computed using the sentinels 0 and n+1.
    #
    # For a pair (v, v+1) we can compute the count of subarrays that contain at least one occurrence of v or v+1 in a similar way
    # by taking the sorted union of the positions for v and v+1.
    # Then by inclusion-exclusion, we get:
    #   count(v and v+1) = count(v) + count(v+1) - count(v union v+1).
    #
    # Finally, summing over all subarrays,
    #   Answer = Sum_{v (1...n)} count(v) - Sum_{v=1}^{n-1} [count(v) + count(v+1) - count(v ∪ (v+1))].
    
    # Total number of subarrays for an array of length n.
    T = n*(n+1)//2
    
    # Build positions for each number 1..n (we use 1-indexing for positions)
    pos = [[] for _ in range(n+1)]
    for i, a in enumerate(A, start=1):
        pos[a].append(i)
        
    # Helper function: given a sorted list "lst" of indices (positions where a value appears),
    # compute the number of subarrays that contain at least one occurrence of that value.
    def count_subarrays(lst):
        if not lst:
            return 0
        total = T
        prev = 0
        gap_sum = 0
        # Go through each occurrence; treat the gap between the previous occurrence (or beginning)
        # and the current occurrence (minus one)
        for p in lst:
            gap = p - prev - 1
            gap_sum += gap*(gap+1)//2
            prev = p
        # Last gap from the last occurrence to the end of the array.
        gap = n - prev
        gap_sum += gap*(gap+1)//2
        return total - gap_sum

    # Compute count(v) for each value v that appears.
    cnt = [0]*(n+1)
    for v in range(1, n+1):
        if pos[v]:
            cnt[v] = count_subarrays(pos[v])
        else:
            cnt[v] = 0

    # (1) Sum_{subarray} (# distinct numbers) equals
    #     the sum over all v of (number of subarrays that contain v)
    distinct_sum = 0
    for v in range(1, n+1):
        distinct_sum += cnt[v]

    # (2) For each consecutive pair (v, v+1), if both occur,
    #     compute count(v and v+1) using:
    #       count(v and v+1) = cnt[v] + cnt[v+1] - count_subarrays( merge(pos[v], pos[v+1]) )
    # where merge(pos[v], pos[v+1]) is the sorted list of positions where either v or v+1 appears.
    
    # Helper function: merge two sorted lists
    def merge_sorted(a, b):
        res = []
        i = j = 0
        la, lb = len(a), len(b)
        while i < la and j < lb:
            if a[i] < b[j]:
                res.append(a[i])
                i += 1
            else:
                res.append(b[j])
                j += 1
        if i < la:
            res.extend(a[i:])
        if j < lb:
            res.extend(b[j:])
        return res
    
    cons_sum = 0
    for v in range(1, n):
        if pos[v] and pos[v+1]:
            merged = merge_sorted(pos[v], pos[v+1])
            union_count = count_subarrays(merged)
            # By inclusion-exclusion:
            cons_sum += cnt[v] + cnt[v+1] - union_count
    
    # The total sum over subarrays of f(L,R) is:
    #   (sum of distinct counts) - (sum of consecutive pairs count)
    ans = distinct_sum - cons_sum
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()