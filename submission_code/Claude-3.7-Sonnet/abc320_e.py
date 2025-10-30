import heapq

def solve():
    N, M = map(int, input().split())
    
    # Initialize noodle counts for each person
    noodle_counts = [0] * (N + 1)
    
    # Event types
    RETURN = 0  # Return events (priority 0) will be processed before noodle events at the same time
    NOODLE = 1
    
    # Initialize events queue
    events = []
    for _ in range(M):
        t, w, s = map(int, input().split())
        heapq.heappush(events, (t, NOODLE, w, s))
    
    # Keep track of who is in the row
    in_row = set(range(1, N + 1))
    
    # Process events
    while events:
        t, event_type, data1, data2 = heapq.heappop(events)
        
        if event_type == NOODLE:  # Noodle event
            w = data1
            s = data2
            
            # Find the person at the front of the row
            front_person = min(in_row) if in_row else None
            
            if front_person:
                noodle_counts[front_person] += w
                in_row.remove(front_person)
                heapq.heappush(events, (t + s, RETURN, front_person, 0))  # Schedule return
        else:  # Return event
            person = data1
            in_row.add(person)
    
    # Print the total noodles each person got
    for i in range(1, N + 1):
        print(noodle_counts[i])

solve()