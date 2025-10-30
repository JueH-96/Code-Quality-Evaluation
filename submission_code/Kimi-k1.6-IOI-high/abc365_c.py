import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    a = list(map(int, input[2:2+n]))
    
    total_sum = sum(a)
    if total_sum <= m:
        print("infinite")
        return
    
    a.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a[i]
    
    max_a = a[-1]
    low, high = 0, max_a
    best = 0
    
    while low <= high:
        mid = (low + high) // 2
        k = bisect.bisect_right(a, mid)
        current_sum = prefix[k] + mid * (n - k)
        if current_sum <= m:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(best)

if __name__ == "__main__":
    main()