def max_distinct_sum(N, A):
    # Step 1: Calculate distinct counts for the left part
    left_distinct = [0] * N
    seen = set()
    
    for i in range(N):
        seen.add(A[i])
        left_distinct[i] = len(seen)
    
    # Step 2: Calculate distinct counts for the right part
    right_distinct = [0] * N
    seen.clear()
    
    for i in range(N-1, -1, -1):
        seen.add(A[i])
        right_distinct[i] = len(seen)
    
    # Step 3: Calculate the maximum sum of distinct counts for valid (i, j)
    max_sum = 0
    
    for j in range(1, N-1):
        # We can choose i from 0 to j-1
        max_left = left_distinct[j-1]
        max_right = right_distinct[j+1]
        max_sum = max(max_sum, max_left + max_right)
    
    return max_sum

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = max_distinct_sum(N, A)
    print(result)

if __name__ == "__main__":
    main()