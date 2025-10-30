import sys
import bisect

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    if sum(A) <= M:
        print("infinite")
        return
    
    A.sort()
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + A[i - 1]
    
    low = 0
    high = A[-1] + 1  # Upper bound is the maximum in A plus one
    
    while low < high:
        mid = (low + high) // 2
        i = bisect.bisect_left(A, mid + 1)
        total = prefix[i] + mid * (N - i)
        if total <= M:
            low = mid + 1
        else:
            high = mid
    
    print(low - 1)

if __name__ == "__main__":
    main()