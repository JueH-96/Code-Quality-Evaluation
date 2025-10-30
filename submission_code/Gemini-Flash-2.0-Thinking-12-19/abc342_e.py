import heapq

def solve():
    n, m = map(int, input().split())
    train_info = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        train_info.append({'l': l, 'd': d, 'k': k, 'c': c, 'a': a, 'b': b})
    
    results = []
    for start_station in range(1, n):
        latest_arrival_time = [-1] * (n + 1)
        latest_arrival_time[start_station] = 0
        pq = [(-0, start_station)]
        
        while pq:
            neg_current_arrival_time, current_station = heapq.heappop(pq)
            current_arrival_time = -neg_current_arrival_time
            if current_arrival_time < latest_arrival_time[current_station]:
                continue
                
            for info in train_info:
                if info['a'] == current_station:
                    l, d, k, c, a_station, b_station = info['l'], info['d'], info['k'], info['c'], info['a'], info['b']
                    for j in range(k - 1, -1, -1):
                        departure_time = l + j * d
                        if departure_time >= current_arrival_time:
                            arrival_time = departure_time + c
                            next_station = b_station
                            if arrival_time > latest_arrival_time[next_station]:
                                latest_arrival_time[next_station] = arrival_time
                                heapq.heappush(pq, (-arrival_time, next_station))
                            break
                            
        result = latest_arrival_time[n]
        if result == -1:
            results.append("Unreachable")
        else:
            results.append(str(result))
            
    print('
'.join(results))

if __name__ == '__main__':
    solve()