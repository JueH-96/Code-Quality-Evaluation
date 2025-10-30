# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    index += 1
    A = list(map(int, input[index:index + N]))
    index += N
    Q = int(input[index])
    index += 1
    queries = []
    for _ in range(Q):
        l = int(input[index])
        r = int(input[index + 1])
        queries.append((l, r))
        index += 2

    results = []
    for l, r in queries:
        start = bisect.bisect_left(A, l)
        end = bisect.bisect_right(A, r)
        total_sleep = 0
        for i in range(max(1, start - start % 2), end - end % 2, 2):
            sleep_start = max(A[i], l)
            sleep_end = min(A[i + 1], r)
            total_sleep += sleep_end - sleep_start
        results.append(total_sleep)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()