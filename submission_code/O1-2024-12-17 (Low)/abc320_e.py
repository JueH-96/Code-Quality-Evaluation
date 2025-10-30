import sys
import heapq

def main():
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    events = []
    
    # Read the event data
    idx = 2
    for _ in range(M):
        T = int(input_data[idx]); W = int(input_data[idx+1]); S = int(input_data[idx+2])
        idx += 3
        events.append((T, W, S))
    
    # events are given in strictly increasing T order, so we can use them as is
    # Data structures:
    # present_minheap: min-heap of currently present person IDs
    # reentry_minheap: min-heap of (reentry_time, person_id) for those who are out
    present_minheap = list(range(1, N+1))
    heapq.heapify(present_minheap)
    reentry_minheap = []
    # To store total noodles gotten by each person (index: person_id-1)
    total_noodles = [0]*N
    
    # Process events in ascending time
    idx_event = 0
    for (T_i, W_i, S_i) in events:
        # First, add back everyone whose reentry_time <= T_i
        while reentry_minheap and reentry_minheap[0][0] <= T_i:
            reentry_time, person_id = heapq.heappop(reentry_minheap)
            # Person re-enters
            heapq.heappush(present_minheap, person_id)
        
        # Now, if present_minheap is not empty, let smallest ID get W_i
        if present_minheap:
            person_id = heapq.heappop(present_minheap)
            total_noodles[person_id - 1] += W_i
            # That person steps out until T_i + S_i
            reentry_time = T_i + S_i
            heapq.heappush(reentry_minheap, (reentry_time, person_id))
        # else: no one gets noodles
    
    # After processing all events, we don't need further changes because no more noodles flow
    # Output the result
    print("
".join(map(str, total_noodles)))

# Ensure main is called
if __name__ == "__main__":
    main()