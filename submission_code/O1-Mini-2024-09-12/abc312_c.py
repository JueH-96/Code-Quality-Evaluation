import sys
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    A.sort()
    B.sort()
    
    def can(X):
        sellers = bisect.bisect_right(A, X)
        buyers = M - bisect.bisect_left(B, X)
        return sellers >= buyers
    
    low = 1
    high = max(A + B) + 1  # Ensure the upper bound covers all possible X
    
    while low < high:
        mid = (low + high) // 2
        if can(mid):
            high = mid
        else:
            low = mid + 1
    
    print(low)

if __name__ == "__main__":
    main()