import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    sum_total = sum(a)
    
    if sum_total <= m:
        print("infinite")
        return
    
    a.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + a[i]
    
    low = 0
    high = 10**18
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        k = bisect.bisect_right(a, mid)
        current_sum = prefix[k] + (n - k) * mid
        
        if current_sum <= m:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == "__main__":
    main()