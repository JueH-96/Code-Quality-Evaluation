import heapq

def solve(n, intervals):
    intervals.sort()
    end_times = []
    print_count, last_print_time = 0, -1

    for start, end in intervals:
        if start > last_print_time + 1:
            if end_times:
                end_earliest = heapq.heappop(end_times)
                if end_earliest <= start:
                    print_count += 1
                    last_print_time = end_earliest
                    heapq.heappush(end_times, end)
                else:
                    heapq.heappush(end_times, start)
            else:
                heapq.heappush(end_times, end)
                print_count += 1
                last_print_time = start

        elif end > last_print_time + 1:
            heapq.heappush(end_times, end)

        while end_times and end_times[0] <= last_print_time + 1:
            heapq.heappop(end_times)

        if end_times and end_times[0] <= start:
            end_earliest = heapq.heappop(end_times)
            print_count += 1
            last_print_time = end_earliest
            heapq.heappush(end_times, end)
        elif start > last_print_time + 1:
            print_count += 1
            last_print_time = start

    while end_times:
        end_earliest = heapq.heappop(end_times)
        if end_earliest > last_print_time + 1:
            print_count += 1
            last_print_time = end_earliest

    return print_count

n = int(input())
intervals = [tuple(map(int, input().split())) for _ in range(n)]
print(solve(n, intervals))