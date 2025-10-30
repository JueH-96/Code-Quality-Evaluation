# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Step 1: Calculate prefix distinct counts
    prefix_distinct_count = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        prefix_distinct_count[i] = len(seen)
    
    # Step 2: Calculate suffix distinct counts
    suffix_distinct_count = [0] * N
    seen = set()
    for i in range(N-1, -1, -1):
        seen.add(A[i])
        suffix_distinct_count[i] = len(seen)
    
    # Step 3: Calculate the maximum sum of distinct counts
    max_sum = 0
    for i in range(N-1):
        # Split between i and i+1
        current_sum = prefix_distinct_count[i] + suffix_distinct_count[i+1]
        max_sum = max(max_sum, current_sum)
    
    print(max_sum)