import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    sum_A = sum(A)
    if sum_A <= M:
        print("infinite")
        return
    
    A.sort()
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    low = 0
    high = A[-1]
    ans = -1
    
    while low <= high:
        mid = (low + high) // 2
        k = bisect.bisect_right(A, mid)
        total = prefix[k] + mid * (N - k)
        if total <= M:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(ans)

if __name__ == '__main__':
    main()