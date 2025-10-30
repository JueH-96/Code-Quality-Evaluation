import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    meds = []
    max_a = 0
    index = 2
    for i in range(n):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        meds.append((a, b))
        if a > max_a:
            max_a = a

    sorted_meds = sorted(meds, key=lambda x: x[0])
    a_sorted = [m[0] for m in sorted_meds]
    b_sorted = [m[1] for m in sorted_meds]
    
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + b_sorted[i]
    
    low = 1
    high = max_a + 1
    ans = high
    while low <= high:
        mid = (low + high) // 2
        idx = bisect.bisect_left(a_sorted, mid)
        total_pills = suffix_sum[idx]
        
        if total_pills <= k:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
            
    print(ans)

if __name__ == "__main__":
    main()