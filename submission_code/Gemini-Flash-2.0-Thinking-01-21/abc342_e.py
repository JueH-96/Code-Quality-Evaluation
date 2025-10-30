import heapq

def solve():
    n, m = map(int, input().split())
    train_info = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        train_info.append({'l': l, 'd': d, 'k': k, 'c': c, 'a': a, 'b': b})
    
    results = []
    for start_station in range(1, n):
        latest_arrival_time = [-float('inf')] * (n + 1)
        latest_arrival_time[start_station] = 0
        pq = [(0, start_station)] # (arrival_time, station)
        
        while pq:
            current_arrival_time, u = heapq.heappop(pq)
            if current_arrival_time < latest_arrival_time[u]:
                continue
                
            for info in train_info:
                if info['a'] == u:
                    for j in range(info['k']):
                        departure_time = info['l'] + j * info['d']
                        arrival_time = departure_time + info['c']
                        v = info['b']
                        if current_arrival_time <= departure_time:
                            if latest_arrival_time[v] < arrival_time:
                                latest_arrival_time[v] = arrival_time
                                heapq.heappush(pq, (arrival_time, v))
                                
        if latest_arrival_time[n] == -float('inf'):
            results.append("Unreachable")
        else:
            results.append(str(latest_arrival_time[n]))
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()