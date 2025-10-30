import math
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    S = float(input[idx]); idx += 1
    T = float(input[idx]); idx += 1

    segments = []
    for _ in range(N):
        A = float(input[idx]); idx += 1
        B = float(input[idx]); idx += 1
        C = float(input[idx]); idx += 1
        D = float(input[idx]); idx += 1
        segments.append(((A, B), (C, D)))

    heap = []
    heapq.heappush(heap, (0.0, 0, (0.0, 0.0)))  # (time, bitmask, position)

    visited = dict()
    initial_key = (0, (0.0, 0.0))
    visited[initial_key] = 0.0

    target = (1 << N) - 1

    while heap:
        current_time, bitmask, current_pos = heapq.heappop(heap)

        if bitmask == target:
            print("{0:.15f}".format(current_time))
            return

        key = (bitmask, current_pos)
        if key in visited and visited[key] < current_time:
            continue

        for i in range(N):
            if not (bitmask & (1 << i)):
                point0, point1 = segments[i]
                # Option 1: start at point0, end at point1
                dx = current_pos[0] - point0[0]
                dy = current_pos[1] - point0[1]
                distance_move = math.hypot(dx, dy)
                time_move = distance_move / S
                distance_emit = math.hypot(point1[0] - point0[0], point1[1] - point0[1])
                time_emit = distance_emit / T
                new_time = current_time + time_move + time_emit
                new_pos = (point1[0], point1[1])
                new_bitmask = bitmask | (1 << i)
                key_new = (new_bitmask, new_pos)
                if key_new not in visited or new_time < visited.get(key_new, float('inf')):
                    visited[key_new] = new_time
                    heapq.heappush(heap, (new_time, new_bitmask, new_pos))
                # Option 2: start at point1, end at point0
                dx = current_pos[0] - point1[0]
                dy = current_pos[1] - point1[1]
                distance_move = math.hypot(dx, dy)
                time_move = distance_move / S
                time_emit = distance_emit / T
                new_time2 = current_time + time_move + time_emit
                new_pos2 = (point0[0], point0[1])
                key_new2 = (new_bitmask, new_pos2)
                if key_new2 not in visited or new_time2 < visited.get(key_new2, float('inf')):
                    visited[key_new2] = new_time2
                    heapq.heappush(heap, (new_time2, new_bitmask, new_pos2))

    print(0.0)

if __name__ == '__main__':
    main()