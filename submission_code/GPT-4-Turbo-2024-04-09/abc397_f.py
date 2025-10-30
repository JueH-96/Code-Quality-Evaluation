import sys
from collections import defaultdict

def read_ints():
    return map(int, sys.stdin.read().split())

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N < 3:
        print(0)
        return
    
    # Calculate distinct counts for the prefix
    prefix_distinct_counts = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        prefix_distinct_counts[i] = len(seen)
    
    # Calculate distinct counts for the suffix
    suffix_distinct_counts = [0] * N
    seen = set()
    for i in range(N-1, -1, -1):
        seen.add(A[i])
        suffix_distinct_counts[i] = len(seen)
    
    # We need to calculate the maximum sum of distinct counts for all possible splits
    max_sum = 0
    
    # We use a sliding window approach to calculate the distinct elements in the middle segment
    for i in range(1, N-1):
        # For each possible i (end of the first segment), we need to find the best j (end of the second segment)
        # such that i < j and j <= N-2
        # We use two pointers to maintain a window from i+1 to j and calculate the distinct count in this window
        middle_distinct_count = defaultdict(int)
        distinct_in_middle = 0
        
        for j in range(i+1, N):
            # Add A[j-1] to the middle segment
            if middle_distinct_count[A[j-1]] == 0:
                distinct_in_middle += 1
            middle_distinct_count[A[j-1]] += 1
            
            # Calculate the sum of distinct counts for the current split
            if j < N-1:  # Ensure there's room for the third segment
                current_sum = prefix_distinct_counts[i-1] + distinct_in_middle + suffix_distinct_counts[j]
                max_sum = max(max_sum, current_sum)
    
    print(max_sum)

if __name__ == "__main__":
    main()