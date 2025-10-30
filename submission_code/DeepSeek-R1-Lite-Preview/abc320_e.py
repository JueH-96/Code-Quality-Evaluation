import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    events = []
    idx = 2
    for _ in range(M):
        T_i = int(data[idx])
        W_i = int(data[idx + 1])
        S_i = int(data[idx + 2])
        events.append((T_i, W_i, S_i))
        idx += 3
    
    # Available people priority queue, ordered by position
    available = list(range(1, N+1))
    heapq.heapify(available)
    
    # Out queue, ordered by return time
    out_queue = []
    
    # Total noodles each person gets
    total_noodles = [0] * (N + 1)
    
    # Process events in order
    for event in events:
        T_i, W_i, S_i = event
        
        # Bring back people who have returned by T_i
        while out_queue and out_queue[0][0] <= T_i:
            return_time, pos = heapq.heappop(out_queue)
            heapq.heappush(available, pos)
        
        # Assign noodles to the first available person
        if available:
            pos = heapq.heappop(available)
            total_noodles[pos] += W_i
            new_return_time = T_i + S_i
            heapq.heappush(out_queue, (new_return_time, pos))
    
    # Print the total noodles for each person
    for i in range(1, N+1):
        print(total_noodles[i])

if __name__ == '__main__':
    main()