# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq

    input = sys.stdin.read().split()
    N = int(input[0])
    events = []

    for i in range(N):
        T_i = int(input[2 * i + 1])
        D_i = int(input[2 * i + 2])
        events.append((T_i, 1, i))  # 1 indicates product enters
        events.append((T_i + D_i, -1, i))  # -1 indicates product leaves

    events.sort()

    available_times = []
    last_print_time = -1
    printed_count = 0

    for time, event_type, product_id in events:
        if event_type == -1:
            # Product leaves, remove its entry from the heap if it's still there
            while available_times and available_times[0] < time:
                heapq.heappop(available_times)
        else:
            # Product enters, add its print time to the heap
            if last_print_time + 1 <= time:
                heapq.heappush(available_times, time)

        # Try to print on a product if possible
        if available_times and available_times[0] <= time:
            print_time = heapq.heappop(available_times)
            if print_time >= last_print_time + 1:
                last_print_time = print_time
                printed_count += 1

    print(printed_count)

if __name__ == "__main__":
    main()