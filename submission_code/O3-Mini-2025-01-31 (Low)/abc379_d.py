def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    if not data:
        return

    Q = int(data[0])
    # global_time represents the cumulative days waited.
    global_time = 0
    # plants list will store the "time offset" when each plant was planted.
    # When a plant is added, its effective height is (global_time - offset).
    plants = []
    output = []
    index = 1

    for _ in range(Q):
        typ = data[index]
        index += 1

        if typ == "1":
            # Query type 1: Plant a new flower pot.
            # The plant is initially of height 0, so we record the current global_time as its offset.
            bisect.insort(plants, global_time)
        elif typ == "2":
            # Query type 2: Wait for T days.
            T = int(data[index])
            index += 1
            global_time += T
        elif typ == "3":
            # Query type 3: Harvest all plants with height >= H.
            H = int(data[index])
            index += 1
            # For a plant planted at offset x, its height becomes global_time - x.
            # Harvest plants with global_time - x >= H => x <= global_time - H.
            # Find the index of the rightmost element <= global_time - H.
            threshold = global_time - H
            # bisect_right returns the insertion point where threshold would be inserted to keep sorted order.
            pos = bisect.bisect_right(plants, threshold)
            # The number of plants satisfying the condition is pos.
            output.append(str(pos))
            # Remove those plants from the list.
            plants = plants[pos:]

    sys.stdout.write("
".join(output))


if __name__ == "__main__":
    main()