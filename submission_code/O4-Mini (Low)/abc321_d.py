import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N, M, P = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Sort side dish prices and build prefix sums
    B.sort()
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i+1] = prefix[i] + B[i]
    
    # For binary search
    import bisect
    
    total = 0
    for a in A:
        # We want all B[j] < P - a
        # bisect_left finds first index where B[idx] >= P - a
        threshold = P - a
        k = bisect.bisect_left(B, threshold)
        # For those k items, sum of (a + B[j])
        sum_small = k * a + prefix[k]
        # For the remaining M - k items, price is P
        sum_large = (M - k) * P
        total += sum_small + sum_large
    
    print(total)

if __name__ == "__main__":
    main()