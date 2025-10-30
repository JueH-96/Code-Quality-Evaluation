def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Precompute distinct counts from the left
    left_distinct = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        left_distinct[i] = len(seen)
    
    # Precompute distinct counts from the right
    right_distinct = [0] * N
    seen = set()
    for i in range(N-1, -1, -1):
        seen.add(A[i])
        right_distinct[i] = len(seen)
    
    # Calculate the maximum sum of distinct counts
    max_distinct_sum = 0
    for i in range(1, N-1):
        for j in range(i+1, N):
            current_sum = left_distinct[i-1] + (right_distinct[i] - right_distinct[j]) + right_distinct[j]
            max_distinct_sum = max(max_distinct_sum, current_sum)
    
    print(max_distinct_sum)