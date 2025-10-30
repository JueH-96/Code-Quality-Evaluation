import heapq

def solve():
    n, m = map(int, input().split())
    trains = []
    for _ in range(m):
        l, d, k, c, a, b = map(int, input().split())
        trains.append((l, d, k, c, a, b))

    def get_trains_at_time(time, start_station):
        available_trains = []
        for l, d, k, c, a, b in trains:
            if a == start_station:
                for i in range(k):
                    departure_time = l + i * d
                    if departure_time >= time:
                        available_trains.append((departure_time, c, a, b))
        return available_trains

    def find_max_arrival_time(start_station):
        max_arrival_time = float('-inf')
        
        q = [(0, start_station, 0)] # (arrival_time, current_station, start_time)
        visited = set()
        
        while q:
            arrival_time, current_station, start_time = heapq.heappop(q)
            
            if (current_station, start_time) in visited:
                continue
            visited.add((current_station, start_time))
            
            if current_station == n:
                max_arrival_time = max(max_arrival_time, arrival_time)
                continue
            
            available_trains = get_trains_at_time(start_time, current_station)
            
            for departure_time, travel_time, _, next_station in available_trains:
                next_arrival_time = departure_time + travel_time
                heapq.heappush(q, (next_arrival_time, next_station, next_arrival_time))
        
        return max_arrival_time

    for start_station in range(1, n):
        result = find_max_arrival_time(start_station)
        if result == float('-inf'):
            print("Unreachable")
        else:
            print(result)

solve()