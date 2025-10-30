import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    total_sum = sum(A)
    if total_sum <= M:
        print("infinite")
        return
        
    A.sort()
    prefix = [0] * (n+1)
    for i in range(1, n+1):
        prefix[i] = prefix[i-1] + A[i-1]
        
    low = 0
    high = A[-1]
    
    def total_subsidy(x):
        idx = bisect.bisect_left(A, x)
        count_ge = n - idx
        sum_lt = prefix[idx]
        return count_ge * x + sum_lt
        
    while low <= high:
        mid = (low + high) // 2
        sub_mid = total_subsidy(mid)
        if sub_mid <= M:
            low = mid + 1
        else:
            high = mid - 1
            
    print(high)

if __name__ == '__main__':
    main()