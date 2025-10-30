import sys
import math
import heapq

def main():
    import sys
    import math
    import heapq

    # Read input
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

    # Collect unique endpoints, including (0,0)
    unique_endpoints = set()
    unique_endpoints.add((0.0, 0.0))
    for seg in segments:
        unique_endpoints.add(seg[0])
        unique_endpoints.add(seg[1])

    # Assign unique indices to endpoints
    endpoint_list = list(unique_endpoints)
    endpoint_indices = {pt: i for i, pt in enumerate(endpoint_list)}
    M = len(endpoint_list)
    start_pos = endpoint_indices[(0.0, 0.0)]

    # Precompute distance matrix
    dist = [[0.0 for _ in range(M)] for _ in range(M)]
    for i in range(M):
        for j in range(M):
            x1, y1 = endpoint_list[i]
            x2, y2 = endpoint_list[j]
            dist[i][j] = math.hypot(x1 - x2, y1 - y2)

    # Precompute printing times for each segment in both directions
    # Since printing time is the same in both directions (distance / T)
    print_time = []
    segment_endpoints = []
    for seg in segments:
        pt1 = endpoint_indices[seg[0]]
        pt2 = endpoint_indices[seg[1]]
        segment_endpoints.append((pt1, pt2))
        seg_dist = dist[pt1][pt2]
        print_time.append(seg_dist / T)

    # Dijkstra's algorithm
    # State: (mask, pos)
    # mask: bitmask of printed segments
    # pos: current position index
    all_printed_mask = (1 << N) - 1
    heap = []
    dp = {}
    initial_mask = 0
    dp[(initial_mask, start_pos)] = 0.0
    heapq.heappush(heap, (0.0, initial_mask, start_pos))

    while heap:
        current_time, mask, pos = heapq.heappop(heap)
        if mask == all_printed_mask:
            # All segments printed, record the time
            min_time = current_time
            continue
        if dp.get((mask, pos), float('inf')) < current_time:
            continue  # Better time already recorded for this state
        for seg_idx in range(N):
            if not (mask & (1 << seg_idx)):
                # Segment not yet printed
                pt1, pt2 = segment_endpoints[seg_idx]
                # Option 1: start from pt1 to pt2
                move_time1 = dist[pos][pt1] / S
                new_time1 = current_time + move_time1 + print_time[seg_idx]
                new_mask1 = mask | (1 << seg_idx)
                new_pos1 = pt2
                if dp.get((new_mask1, new_pos1), float('inf')) > new_time1:
                    dp[(new_mask1, new_pos1)] = new_time1
                    heapq.heappush(heap, (new_time1, new_mask1, new_pos1))
                # Option 2: start from pt2 to pt1
                move_time2 = dist[pos][pt2] / S
                new_time2 = current_time + move_time2 + print_time[seg_idx]
                new_mask2 = mask | (1 << seg_idx)
                new_pos2 = pt1
                if dp.get((new_mask2, new_pos2), float('inf')) > new_time2:
                    dp[(new_mask2, new_pos2)] = new_time2
                    heapq.heappush(heap, (new_time2, new_mask2, new_pos2))

    # Find the minimum time among all states where all segments are printed
    min_time = float('inf')
    for (mask, pos), time in dp.items():
        if mask == all_printed_mask:
            if time < min_time:
                min_time = time
    print("{0:.15f}".format(min_time))

if __name__ == '__main__':
    main()