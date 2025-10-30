import heapq

def main():
    n = int(input())
    coords = [tuple(map(int, input().split())) for _ in range(n)]
    
    def dist(i, j):
        dx = coords[j][0] - coords[i][0]
        dy = coords[j][1] - coords[i][1]
        return (dx**2 + dy**2)**0.5
    
    heap = []
    heapq.heappush(heap, (0.0, 0, 0))  # (priority, current, skips)
    visited = {}
    target = n - 1
    
    while heap:
        current_priority, current, skips = heapq.heappop(heap)
        if current == target:
            print("{0:.10f}".format(current_priority))
            return
        if (current, skips) in visited and visited[(current, skips)] <= current_priority:
            continue
        visited[(current, skips)] = current_priority
        max_j = min(current + 40, target)
        for j in range(current + 1, max_j + 1):
            k = j - current - 1
            new_skips = skips + k
            if new_skips > 40:
                continue
            d = dist(current, j)
            penalty_old = 2 ** (skips - 1) if skips > 0 else 0
            new_distance = current_priority - penalty_old + d
            penalty_new = 2 ** (new_skips - 1) if new_skips > 0 else 0
            new_priority = new_distance + penalty_new
            key = (j, new_skips)
            if key not in visited or new_priority < visited.get(key, float('inf')):
                heapq.heappush(heap, (new_priority, j, new_skips))
    
if __name__ == "__main__":
    main()