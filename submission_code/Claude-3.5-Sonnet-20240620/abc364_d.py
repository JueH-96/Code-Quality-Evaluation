# YOUR CODE HERE
import sys
from bisect import bisect_left

def main():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    A.sort()

    for _ in range(Q):
        b, k = map(int, input().split())
        
        # Find the index where b would be inserted in the sorted A
        idx = bisect_left(A, b)
        
        # Create a list of distances
        distances = []
        
        # Check k elements to the left and right of idx
        left = max(0, idx - k)
        right = min(N, idx + k)
        
        for i in range(left, right):
            distances.append(abs(A[i] - b))
        
        # Sort distances and get the k-th element
        distances.sort()
        print(distances[k-1])

if __name__ == "__main__":
    main()