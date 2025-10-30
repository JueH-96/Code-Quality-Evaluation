def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    
    T = [0]*M
    W = [0]*M
    S = [0]*M
    
    idx = 2
    for i in range(M):
        T[i] = int(input_data[idx]); 
        W[i] = int(input_data[idx+1]); 
        S[i] = int(input_data[idx+2])
        idx += 3
    
    # We'll create 2M events: (time, type, i)
    # type = 0 for return, 1 for noodle
    # We sort by time ascending, and if tie, type ascending (so return events first)
    events = []
    for i in range(M):
        # Return event
        events.append((T[i] + S[i], 0, i))
        # Noodle event
        events.append((T[i], 1, i))
    
    # Sort by (time, type)
    events.sort(key=lambda x: (x[0], x[1]))
    
    # Min-heap of currently present people (by ID)
    present = list(range(1, N+1))
    heapq.heapify(present)
    
    # Record of who got noodles in each of the M noodle events
    winner = [None]*M
    
    # Amount of noodles each person gets
    result = [0]*N
    
    # Process events in sorted order
    i_ev = 0
    len_ev = len(events)
    while i_ev < len_ev:
        t, etype, idx_m = events[i_ev]
        if etype == 0:
            # Return event for the winner of the noodle event idx_m
            p = winner[idx_m]
            if p is not None:
                # Re-insert p to heap
                heapq.heappush(present, p)
        else:
            # Noodle event
            if present:
                p = heapq.heappop(present)
                result[p-1] += W[idx_m]
                winner[idx_m] = p
        i_ev += 1
    
    # Output results
    print("
".join(map(str, result)))

# Call main
main()