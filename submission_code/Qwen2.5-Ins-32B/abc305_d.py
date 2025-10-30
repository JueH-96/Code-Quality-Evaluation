import bisect

def solve_sleep_log(N, A, Q, queries):
    # Create a list to store the cumulative sleep time up to each minute
    sleep_time = [0] * (A[-1] + 1)
    
    # Fill the sleep_time list with the sleep intervals
    for i in range(1, (N // 2) + 1):
        start, end = A[2 * i - 1], A[2 * i]
        for j in range(start, end):
            sleep_time[j] = 1
    
    # Convert sleep_time to cumulative sum
    for i in range(1, len(sleep_time)):
        sleep_time[i] += sleep_time[i - 1]
    
    # Answer each query
    results = []
    for l, r in queries:
        if l == 0:
            results.append(sleep_time[r])
        else:
            results.append(sleep_time[r] - sleep_time[l - 1])
    
    return results

# Read input
N = int(input())
A = list(map(int, input().split()))
Q = int(input())
queries = [tuple(map(int, input().split())) for _ in range(Q)]

# Solve the problem
results = solve_sleep_log(N, A, Q, queries)

# Print the results
for result in results:
    print(result)