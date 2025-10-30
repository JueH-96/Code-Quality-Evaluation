def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])

    # cur_time will track the "global offset" of plant growth
    cur_time = 0

    # A min-heap storing the "creation offset" for each plant
    # So each plant's current height = cur_time - creation_offset
    plants = []

    # We'll read from input_data using an index pointer:
    idx = 1

    # Answers to type-3 queries
    answers = []

    for _ in range(Q):
        query_type = int(input_data[idx])
        idx += 1

        if query_type == 1:
            # Create a new plant with height 0 at the current time
            # Store the creation offset
            heapq.heappush(plants, cur_time)

        elif query_type == 2:
            # Wait T days => increase cur_time by T
            T = int(input_data[idx])
            idx += 1
            cur_time += T

        else:
            # query_type == 3
            H = int(input_data[idx])
            idx += 1

            # Harvest all plants whose height >= H
            # height >= H => (cur_time - creation_offset) >= H => creation_offset <= cur_time - H
            harvest_count = 0
            boundary = cur_time - H

            while plants and plants[0] <= boundary:
                heapq.heappop(plants)
                harvest_count += 1

            answers.append(harvest_count)

    # Print all answers to type-3 queries
    print('
'.join(map(str, answers)))

# Do not forget to call main()
if __name__ == "__main__":
    main()