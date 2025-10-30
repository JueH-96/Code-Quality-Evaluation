def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])

    # A min-heap for storing "insertion offsets" of the plants.
    # offset_plant = current offset when the plant was inserted
    # Real height at any time is (current_offset - offset_plant).
    heap = []
    offset = 0
    idx = 1  # index to traverse input_data

    out = []
    for _ in range(Q):
        query_type = int(input_data[idx]); idx += 1
        if query_type == 1:
            # Plant a new plant (height 0) => push the current offset
            heapq.heappush(heap, offset)
        elif query_type == 2:
            # Wait T days => offset += T
            T = int(input_data[idx]); idx += 1
            offset += T
        else:
            # Harvest plants with height >= H => offset - offset_plant >= H => offset_plant <= offset - H
            H = int(input_data[idx]); idx += 1
            threshold = offset - H
            count = 0
            # Pop from heap while smallest offset_plant <= threshold
            while heap and heap[0] <= threshold:
                heapq.heappop(heap)
                count += 1
            out.append(str(count))
    
    print("
".join(out))

# Do not forget to call main
if __name__ == "__main__":
    main()