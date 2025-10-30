def main():
    import sys
    import heapq
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Flow events: T_i, W_i, S_i
    T = [0]*M
    W = [0]*M
    S = [0]*M
    
    idx = 2
    for i in range(M):
        T[i] = int(input_data[idx]);     idx += 1
        W[i] = int(input_data[idx]);     idx += 1
        S[i] = int(input_data[idx]);     idx += 1
    
    # active[p] indicates if person p (1-based) is currently in the row
    active = [True]*(N+1)
    # Min-heap for "who is present," keyed by person-ID; we want the smallest ID at the front
    front = []
    # Populate with all people initially
    for p in range(1, N+1):
        heapq.heappush(front, p)
    
    # Min-heap for return events: (return_time, person)
    returns = []
    
    # Amount of noodles each person gets
    ans = [0]*N
    
    # Process each flow event in the order they are given (T is strictly increasing)
    for i in range(M):
        cur_t = T[i]
        # Handle any returns up to time cur_t
        while returns and returns[0][0] <= cur_t:
            ret_time, person = heapq.heappop(returns)
            # Mark the person active again
            active[person] = True
            # Put them back into the "front" structure
            heapq.heappush(front, person)
        
        # Now handle the flow at time cur_t
        # First, pop stale IDs (those marked inactive) from the front heap
        while front and not active[front[0]]:
            heapq.heappop(front)
        
        if front:
            # The person with the smallest ID who is active
            p = heapq.heappop(front)
            # This person gets W[i] noodles
            ans[p-1] += W[i]
            # Temporarily mark them inactive
            active[p] = False
            # Schedule their return
            ret_time = cur_t + S[i]
            heapq.heappush(returns, (ret_time, p))
        # If front is empty, nobody gets the noodles; no further action
    
    # Output the result
    print("
".join(str(x) for x in ans))

# Do not forget to call main().
main()