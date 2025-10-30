def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])
    idx_data = 1

    creation_times = []
    p = 0  # Pointer to the first unharvested plant
    current_time = 0
    answers = []

    for _ in range(Q):
        query_type = input_data[idx_data]
        idx_data += 1

        if query_type == "1":
            # Plant a new one with baseline current_time
            creation_times.append(current_time)

        elif query_type == "2":
            # Wait for T days
            T = int(input_data[idx_data])
            idx_data += 1
            current_time += T

        else:  # query_type == "3"
            H = int(input_data[idx_data])
            idx_data += 1
            # Harvest plants with height >= H => creation_time <= current_time - H
            X = current_time - H
            # Number of plants <= X
            harvest_idx = bisect.bisect_right(creation_times, X)
            harvested = harvest_idx - p
            p = harvest_idx
            answers.append(harvested)

    print("
".join(map(str, answers)))

# Do not forget to call main() at the end
main()