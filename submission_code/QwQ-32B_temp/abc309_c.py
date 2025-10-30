import bisect

def main():
    import sys
    n, k = map(int, sys.stdin.readline().split())
    meds = []
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        meds.append((a, b))
    
    # Sort medicines by a_i in ascending order
    meds.sort()
    sorted_a = [x[0] for x in meds]
    sorted_b = [x[1] for x in meds]
    
    # Compute suffix sums
    suffix_sum = [0] * n
    suffix_sum[-1] = sorted_b[-1]
    for i in range(n-2, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + sorted_b[i]
    
    max_a = sorted_a[-1] if n else 0
    low = 1
    high = max_a + 1
    
    while low < high:
        mid = (low + high) // 2
        pos = bisect.bisect_left(sorted_a, mid)
        total = suffix_sum[pos] if pos < n else 0
        if total > k:
            low = mid + 1
        else:
            high = mid
    print(low)

if __name__ == "__main__":
    main()