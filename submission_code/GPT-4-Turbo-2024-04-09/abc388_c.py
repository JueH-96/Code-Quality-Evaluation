import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    count = 0
    
    # Since the array is already sorted, we can use binary search to find the upper bound
    for i in range(N):
        a = A[i]
        # Find the largest index j such that A[j] >= 2 * a
        j = bisect.bisect_right(A, 2 * a)
        # All elements from i+1 to j-1 are valid mochi B for mochi A
        count += j - i - 1
    
    print(count)

if __name__ == "__main__":
    main()