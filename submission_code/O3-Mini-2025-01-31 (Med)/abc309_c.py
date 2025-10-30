def main():
    import sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    
    # Read medicines: each medicine is given as (a_i, b_i)
    meds = []
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        meds.append((a, b))
    
    # We need to quickly compute for any given day d the total pills taken
    # on that day: for each medicine with a_i >= d, add b_i.
    # Since the function is non-increasing in d, we can binary search for the first day
    # when the total becomes <= K.
    
    # Sort medicines by a_i in increasing order.
    meds.sort(key=lambda x: x[0])
    a_list = [item[0] for item in meds]
    
    # Precompute the suffix-sum for b_i.
    # suffix[i] stores the sum of b_i for all medicines from index i to the end.
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + meds[i][1]
    
    # Define function to compute the total pills taken on day d.
    # We find the first position in a_list such that a_i >= d.
    def pills(d):
        idx = bisect.bisect_left(a_list, d)
        return suffix[idx]
    
    # Set the search range.
    # On day (max(a_i)+1), no medicine is taken (i.e., 0 pills),
    # which is guaranteed to be <= K since K >= 0.
    lo = 1
    hi = a_list[-1] + 1
    ans = hi
    
    # Binary search for the smallest day d such that pills(d) <= K.
    while lo <= hi:
        mid = (lo + hi) // 2
        if pills(mid) <= K:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Output the answer.
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()