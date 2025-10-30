# YOUR CODE HERE
def calculate_overlap(sleep_start, sleep_end, query_start, query_end):
    """Calculate overlap between two intervals"""
    overlap_start = max(sleep_start, query_start)
    overlap_end = min(sleep_end, query_end)
    return max(0, overlap_end - overlap_start)

# Read input
N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Process each query
for _ in range(Q):
    l, r = map(int, input().split())
    
    total_sleep = 0
    
    # Check each sleep session
    for i in range(1, N, 2):  # i = 1, 3, 5, ... (even indices in 0-based)
        sleep_start = A[i]      # Even index (0-based) = went to bed
        sleep_end = A[i + 1]    # Odd index (0-based) = woke up
        
        # Calculate overlap between sleep session and query interval
        overlap = calculate_overlap(sleep_start, sleep_end, l, r)
        total_sleep += overlap
    
    print(total_sleep)