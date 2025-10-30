import heapq

def solve():
    n, a, b, c = map(int, input().split())
    d = []
    for _ in range(n):
        d.append(list(map(int, input().split())))

    def dijkstra(start_node, use_car):
        dist = {node: float('inf') for node in range(n)}
        dist[start_node] = 0
        pq = [(0, start_node)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_dist > dist[current_node]:
                continue

            for neighbor in range(n):
                if neighbor == current_node:
                    continue

                if use_car:
                    weight = d[current_node][neighbor] * a
                else:
                    weight = d[current_node][neighbor] * b + c

                new_dist = current_dist + weight

                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        return dist

    car_dist = dijkstra(0, True)
    
    train_dist = dijkstra(0, False)
    
    min_dist = float('inf')
    
    # Option 1: Only car
    min_dist = min(min_dist, car_dist[n-1])
    
    # Option 2: Only train
    #min_dist = min(min_dist, train_dist[n-1])
    
    # Option 3: Car then train
    
    car_to_node = car_dist
    
    for switch_node in range(n):
        train_dist_from_switch = dijkstra(switch_node, False)
        min_dist = min(min_dist, car_to_node[switch_node] + train_dist_from_switch[n-1])
        
    print(int(min_dist))

solve()