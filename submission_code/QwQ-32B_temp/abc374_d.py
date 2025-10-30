import heapq
import sys

def main():
    N, S, T = map(int, sys.stdin.readline().split())
    segments_info = []
    points = [(0.0, 0.0)]  # List of tuples (x, y) as floats
    point_to_index = {(0.0, 0.0): 0}

    for _ in range(N):
        A, B, C, D = map(int, sys.stdin.readline().split())
        p1 = (A, B)
        p2 = (C, D)
        for p in [p1, p2]:
            if p not in point_to_index:
                idx = len(points)
                point_to_index[p] = idx
                points.append((float(p[0]), float(p[1])))
        idx1 = point_to_index[p1]
        idx2 = point_to_index[p2]
        dx = A - C
        dy = B - D
        length = (dx**2 + dy**2)**0.5
        segments_info.append((idx1, idx2, length))

    P = len(points)
    max_mask = 1 << N
    INF = float('inf')
    min_time = [[INF] * max_mask for _ in range(P)]
    min_time[0][0] = 0.0  # Starting at (0,0), mask 0

    heap = []
    heapq.heappush(heap, (0.0, 0, 0))

    answer = INF
    while heap:
        current_time, current_p, current_mask = heapq.heappop(heap)
        if current_mask == ( (1 << N) - 1 ):
            answer = current_time
            break
        if current_time > min_time[current_p][current_mask]:
            continue
        for i in range(N):
            if (current_mask & (1 << i)) != 0:
                continue  # Already processed this segment
            seg = segments_info[i]
            idx1, idx2, length = seg
            for start_p, end_p in [(idx1, idx2), (idx2, idx1)]:
                # Calculate distance from current position to start_p
                x1, y1 = points[current_p]
                x2, y2 = points[start_p]
                dx = x1 - x2
                dy = y1 - y2
                distance = (dx**2 + dy**2)**0.5
                time_move = distance / S
                time_print = length / T
                new_time = current_time + time_move + time_print
                new_mask = current_mask | (1 << i)
                new_p = end_p
                if new_time < min_time[new_p][new_mask]:
                    min_time[new_p][new_mask] = new_time
                    heapq.heappush(heap, (new_time, new_p, new_mask))
    
    print("{0:.12f}".format(answer))

if __name__ == "__main__":
    main()