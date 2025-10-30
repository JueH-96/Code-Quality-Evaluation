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
    
    trains = [[] for _ in range(N + 1)]
    for _ in range(M):
        l = int(data[idx])
        idx += 1
        d = int(data[idx])
        idx += 1
        k = int(data[idx])
        idx += 1
        c = int(data[idx])
        idx += 1
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        trains[A].append((l, d, k, c, B))
    
    max_time = [-1] * (N + 1)
    heap = []
    
    # Compute initial max_time[N]
    max_nt = -1
    for train in trains[N]:
        pass  # Wait, no, trains are stored by A_i. So to find all trains that end at N, we need to look at all trains where B == N.
    # Wait, no. The 'trains' list is indexed by A_i. So to find all trains that end at N, we need to iterate through all trains and check if B == N.
    # This is inefficient, but for the sake of correctness:
    max_nt = -1
    for A in range(1, N + 1):
        for train in trains[A]:
            _, _, _, _, B = train
            if B == N:
                l, d, k, c, _ = train
                last_t = l + (k - 1) * d
                arrival = last_t + c
                if arrival > max_nt:
                    max_nt = arrival
    if max_nt == -1:
        for i in range(1, N):
            print("Unreachable")
        return
    
    max_time[N] = max_nt
    heapq.heappush(heap, (-max_nt, N))
    
    while heap:
        current = heapq.heappop(heap)
        current_time = -current[0]
        current_station = current[1]
        
        if current_time < max_time[current_station]:
            continue
        
        for train in trains[current_station]:
            l, d, k, c, B = train
            # Compute the maximum t <= current_time
            if l > current_time:
                continue
            m = (current_time - l) // d
            if m < 0:
                continue
            m = min(m, k - 1)
            if m < 0:
                continue
            t = l + m * d
            arrival = t + c
            if arrival > max_time[B]:
                max_time[B] = arrival
                heapq.heappush(heap, (-arrival, B))
    
    for i in range(1, N):
        if max_time[i] == -1:
            print("Unreachable")
        else:
            print(max_time[i])

if __name__ == '__main__':
    main()