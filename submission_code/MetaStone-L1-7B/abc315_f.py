import math
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    checkpoints = []
    idx = 1
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx+1])
        checkpoints.append((x, y))
        idx += 2
    
    # Precompute distances between all pairs
    dist = {}
    for i in range(N):
        for j in range(i+1, N):
            dx = checkpoints[i][0] - checkpoints[j][0]
            dy = checkpoints[i][1] - checkpoints[j][1]
            dist[(i, j)] = math.hypot(dx, dy)
    
    # Initialize priority queue
    heap = []
    heapq.heappush(heap, (0.0, 0, 0))
    
    # DP dictionary: key is (i, c), value is the minimal sum
    dp = {}
    
    while heap:
        current_sum, i, c = heapq.heappop(heap)
        
        if i >= N:
            penalty = 2 ** max(0, c - 1)
            total_s = current_sum + penalty
            print("{0:.10f}".format(total_s))
            return
        
        if (i, c) in dp and dp[(i, c)] <= current_sum:
            continue
        
        dp[(i, c)] = current_sum
        
        # Option 1: include checkpoint i+1
        if i + 1 < N:
            next_i = i + 1
            new_sum = current_sum + dist[(i, next_i)]
            if (next_i, c) not in dp or new_sum < dp.get((next_i, c), float('inf')):
                heapq.heappush(heap, (new_sum, next_i, c))
        
        # Option 2: skip checkpoint i+1 (if i+1 >=2)
        if i + 1 >= 2:
            prev_i = i - 1
            next_i = i + 1
            new_sum_skip = current_sum + dist.get((prev_i, next_i), 0)
            new_c_skip = c + 1
            if (next_i, new_c_skip) not in dp or new_sum_skip < dp.get((next_i, new_c_skip), float('inf')):
                heapq.heappush(heap, (new_sum_skip, next_i, new_c_skip))
    
    # If heap is empty and no solution found, return something, but problem states it's always possible
    # So, this part is just to handle, but the problem says it's possible.
    print("Error")

if __name__ == "__main__":
    main()