import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    total = sum(A)
    if total <= M:
        print("infinite")
        return
        
    A.sort()
    prefix = [0] * (n+1)
    for i in range(n):
        prefix[i+1] = prefix[i] + A[i]
    
    low, high = 0, A[-1]
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        idx = bisect.bisect_left(A, mid)
        total_subsidy = prefix[idx] + mid * (n - idx)
        if total_subsidy <= M:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
            
    print(ans)

if __name__ == "__main__":
    main()