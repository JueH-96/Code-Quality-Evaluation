def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    chords = []
    idx = 1
    for _ in range(N):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        idx += 2
        # Normalize so that A < B
        if A > B:
            A, B = B, A
        chords.append((A, B))
    
    # Prepare a list of endpoints: (position, type, index)
    # type = +1 for start of chord, -1 for end of chord
    endpoints = []
    for i, (A, B) in enumerate(chords):
        endpoints.append((A, 1, i))
        endpoints.append((B, -1, i))
    
    # Sort endpoints by position; in case of tie, start (+1) before end (-1)
    endpoints.sort(key=lambda x: (x[0], -x[1]))
    
    import heapq
    
    # We'll keep a max-heap of B-values (negative for heapq),
    # and a flag to mark closed chords
    active_heap = []
    closed = [False]*N
    
    # Iterate through endpoints
    for point, t, i in endpoints:
        if t == 1:
            # Starting a chord
            current_B = chords[i][1]
            
            # Pop away chords that are closed
            while active_heap and closed[active_heap[0][1]]:
                heapq.heappop(active_heap)
            
            # If there's still something active, check for intersection
            if active_heap:
                # The top of the heap is the chord with the largest B among open chords
                largest_B, idx_chord = active_heap[0]
                largest_B = -largest_B  # we stored as negative
                # If largest_B > current_B, we have an intersection
                if largest_B > current_B:
                    print("Yes")
                    return
            
            # Add the current chord's B to the active set
            heapq.heappush(active_heap, (-current_B, i))
        else:
            # Ending a chord
            closed[i] = True
    
    # If we never found an intersection
    print("No")

# Do not forget to call main()
main()