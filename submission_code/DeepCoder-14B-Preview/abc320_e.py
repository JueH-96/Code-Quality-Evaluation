import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    events = []
    for _ in range(M):
        T = int(data[idx])
        idx += 1
        W = int(data[idx])
        idx += 1
        S = int(data[idx])
        idx += 1
        events.append((T, W, S))
    
    # Initialize available people as a min-heap based on their original numbers
    available = list(range(1, N + 1))
    heapq.heapify(available)
    
    out_heap = []
    total = [0] * (N + 1)  # 1-based indexing
    
    for T_i, W_i, S_i in events:
        # Add all people who have returned by T_i
        while out_heap and out_heap[0][0] <= T_i:
            return_time, person = heapq.heappop(out_heap)
            heapq.heappush(available, person)
        
        # Assign noodles if possible
        if available:
            person = heapq.heappop(available)
            total[person] += W_i
            heapq.heappush(out_heap, (T_i + S_i, person))
    
    # Output the results
    for i in range(1, N + 1):
        print(total[i])

if __name__ == '__main__':
    main()