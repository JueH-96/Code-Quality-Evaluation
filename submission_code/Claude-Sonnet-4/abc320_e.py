import heapq

N, M = map(int, input().split())

# Store noodle events
events = []
for _ in range(M):
    t, w, s = map(int, input().split())
    events.append((t, w, s))

# Initialize queue with people 1 to N
queue = list(range(1, N + 1))

# Track total noodles for each person
noodles = [0] * (N + 1)  # 1-indexed

# Priority queue for return events: (return_time, person_id, original_position)
return_heap = []

for t, w, s in events:
    # Process all returns that happen at or before time t
    while return_heap and return_heap[0][0] <= t:
        return_time, person_id, original_pos = heapq.heappop(return_heap)
        
        # Find where to insert this person back
        # They should be inserted at their original position relative to people still in queue
        insert_pos = 0
        for i, person_in_queue in enumerate(queue):
            if person_in_queue > original_pos:
                insert_pos = i
                break
        else:
            insert_pos = len(queue)
        
        queue.insert(insert_pos, person_id)
    
    # Give noodles to front person if queue is not empty
    if queue:
        front_person = queue.pop(0)
        noodles[front_person] += w
        
        # Schedule return
        return_time = t + s
        heapq.heappush(return_heap, (return_time, front_person, front_person))

# Output results
for i in range(1, N + 1):
    print(noodles[i])