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
    
    trains = [[] for _ in range(N + 1)]  # trains[b] contains (A, l, d, k, c)
    for _ in range(M):
        l = int(data[idx])
        idx +=1
        d = int(data[idx])
        idx +=1
        k = int(data[idx])
        idx +=1
        c = int(data[idx])
        idx +=1
        A = int(data[idx])
        idx +=1
        B = int(data[idx])
        idx +=1
        trains[B].append( (A, l, d, k, c) )
    
    f = [-1] * (N + 1)
    f[N] = 0
    heap = [ (-f[N], N) ]
    heapq.heapify(heap)
    processed = [False] * (N + 1)
    
    while heap:
        current_f_neg, u = heapq.heappop(heap)
        current_f = -current_f_neg
        if processed[u]:
            continue
        processed[u] = True
        
        for (A, l, d, k, c) in trains[u]:
            max_c = current_f - c
            if max_c < l:
                continue
            # Compute m_max
            numerator = max_c - l
            if numerator < 0:
                m_max = -1
            else:
                m_max = numerator // d
                m_max = min(m_max, k-1)
            
            if m_max < 0:
                continue
            departure_time = l + m_max * d
            if departure_time > max_c:
                m_max -= 1
                if m_max < 0:
                    continue
                departure_time = l + m_max * d
            
            arrival_at_u = departure_time + c
            if arrival_at_u > current_f:
                continue
            
            candidate_f_A = arrival_at_u + current_f
            if candidate_f_A > f[A] or f[A] == -1:
                f[A] = candidate_f_A
                heapq.heappush(heap, (-candidate_f_A, A))
    
    for s in range(1, N):
        if f[s] == -1:
            print("Unreachable")
        else:
            print(f[s])

if __name__ == "__main__":
    main()