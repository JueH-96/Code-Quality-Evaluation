# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    Q = int(input[index + 1])
    index += 2
    R = list(map(int, input[index:index + N]))
    index += N
    queries = list(map(int, input[index:index + Q]))

    R.sort()

    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + R[i]

    results = []
    for X in queries:
        # Find the maximum number of sleighs that can be pulled with X reindeer
        # We need to find the largest k such that prefix_sum[k] <= X
        k = bisect.bisect_right(prefix_sum, X) - 1
        results.append(k)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()