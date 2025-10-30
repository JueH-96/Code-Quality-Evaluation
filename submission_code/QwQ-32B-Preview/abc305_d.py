from bisect import bisect_left

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    queries = list(map(int, data[N+2:N+2+2*Q]))
    
    # Collect all unique times: sleep times and query boundaries
    timeline = A[1:] + [queries[i] for i in range(0, 2*Q, 2)] + [queries[i+1] for i in range(0, 2*Q, 2)]
    timeline = sorted(set(timeline))
    
    # Identify sleep starts and ends
    sleep_starts = set(A[2::2])  # A2, A4, ..., A_{N-1}
    sleep_ends = set(A[1::2])    # A3, A5, ..., A_N
    
    # Build prefix sum array
    prefix = [0] * (len(timeline) + 1)
    in_sleep = False
    previous_time = 0
    for i in range(1, len(timeline)):
        current_time = timeline[i]
        if in_sleep:
            prefix[i] = prefix[i-1] + (current_time - previous_time)
        else:
            prefix[i] = prefix[i-1]
        if current_time in sleep_starts:
            in_sleep = True
        elif current_time in sleep_ends:
            in_sleep = False
        previous_time = current_time
    # Copy the last prefix sum value to the end
    prefix[len(timeline)] = prefix[len(timeline)-1]
    
    # Answer the queries
    for q in range(Q):
        l = queries[2*q]
        r = queries[2*q+1]
        # Find positions in timeline
        li = bisect_left(timeline, l)
        ri = bisect_left(timeline, r)
        # Compute prefix[ri] - prefix[li]
        answer = prefix[ri] - prefix[li]
        print(answer)

if __name__ == "__main__":
    main()