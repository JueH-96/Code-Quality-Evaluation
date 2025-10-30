import heapq

def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())
    start_node = (sx, sy)
    target_node = (tx, ty)
    
    min_x = min(sx, tx) - 5
    max_x = max(sx, tx) + 5
    min_y = min(sy, ty) - 5
    max_y = max(sy, ty) + 5
    
    distances = {}
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            distances[(i, j)] = float('inf')
    distances[start_node] = 0
    
    pq = [(0, start_node)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue
        if current_node == target_node:
            return current_distance
            
        cx, cy = current_node
        
        # Move to (cx+1, cy)
        neighbor = (cx + 1, cy)
        if min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y:
            cost = 1 - ((cx + cy) % 2)
            new_distance = current_distance + cost
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
                
        # Move to (cx-1, cy)
        neighbor = (cx - 1, cy)
        if min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y:
            cost = (cx + cy) % 2
            new_distance = current_distance + cost
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
                
        # Move to (cx, cy+1)
        neighbor = (cx, cy + 1)
        if min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y:
            cost = 1
            new_distance = current_distance + cost
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
                
        # Move to (cx, cy-1)
        neighbor = (cx, cy - 1)
        if min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y:
            cost = 1
            new_distance = current_distance + cost
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
                
    return -1 # Should not reach here if target is reachable

result = solve()
print(result)