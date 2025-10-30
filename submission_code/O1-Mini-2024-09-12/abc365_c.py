import sys
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    A.sort()
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    total_sum = prefix[N]
    if total_sum <= M:
        print("infinite")
        return
    
    left = 0
    right = A[-1]
    answer = 0
    while left <= right:
        mid = (left + right) // 2
        k = bisect.bisect_right(A, mid)
        subsidy = prefix[k] + mid*(N - k)
        if subsidy <= M:
            answer = mid
            left = mid + 1
        else:
            right = mid -1
    print(answer)

if __name__ == "__main__":
    main()