import sys

def calculate_sleep_time(A, Q, queries):
    """
    Calculate the total sleep time for each query.

    Args:
    A (list): The sleep log.
    Q (int): The number of queries.
    queries (list): The queries.

    Returns:
    list: The total sleep time for each query.
    """
    sleep_intervals = []
    for i in range(1, len(A) // 2 + 1):
        sleep_intervals.append((A[2 * i - 1], A[2 * i]))

    results = []
    for query in queries:
        l, r = query
        total_sleep_time = 0
        for start, end in sleep_intervals:
            if start > r or end < l:
                continue
            if start < l and end > r:
                total_sleep_time += r - l
            elif start < l and end <= r:
                total_sleep_time += end - l
            elif start >= l and end > r:
                total_sleep_time += r - start
            else:
                total_sleep_time += end - start
        results.append(total_sleep_time)

    return results

def main():
    N = int(input())
    A = list(map(int, input().split()))
    Q = int(input())
    queries = [list(map(int, input().split())) for _ in range(Q)]

    results = calculate_sleep_time(A, Q, queries)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()