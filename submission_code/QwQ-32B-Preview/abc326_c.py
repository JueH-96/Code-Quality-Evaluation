import sys
import bisect

def main():
    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    # Sort the array A
    A.sort()
    
    max_count = 0
    for i in range(N):
        # Find the first index j where A[j] >= A[i] + M
        j = bisect.bisect_left(A, A[i] + M)
        # Number of gifts in [A[i], A[i] + M) is j - i
        current_count = j - i
        if current_count > max_count:
            max_count = current_count
    
    print(max_count)

if __name__ == "__main__":
    main()