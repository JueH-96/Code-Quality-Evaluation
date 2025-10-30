import bisect
import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    A.sort()
    B.sort()
    
    total = 0
    j = 0  # Pointer for the current position in A
    
    for b in B:
        # Find the first element in A starting from index j that is >= b
        pos = bisect.bisect_left(A, b, j)
        if pos >= len(A):
            print(-1)
            return
        total += A[pos]
        j = pos + 1  # Move pointer to the next position
    
    print(total)

if __name__ == "__main__":
    main()