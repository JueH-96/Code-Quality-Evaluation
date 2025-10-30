import bisect

def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    total = sum(a)
    if total <= m:
        print("infinite")
        return
    
    a.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + a[i]
    
    low = 0
    high = a[-1]
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        idx = bisect.bisect_right(a, mid)
        current_sum = prefix[idx] + mid * (n - idx)
        if current_sum <= m:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == "__main__":
    main()