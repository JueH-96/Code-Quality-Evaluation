def min_operations(T, test_cases):
    results = []
    for case in test_cases:
        N, P = case
        if P == list(range(1, N+1)):
            results.append(0)
            continue
        # Find the first and last positions where P[i] != i+1
        first = 0
        last = N-1
        while first < N and P[first] == first+1:
            first += 1
        while last >= 0 and P[last] == last+1:
            last -= 1
        # Check if the middle part is already sorted
        if first > last:
            results.append(0)
            continue
        # Check if the entire segment from first to last is already sorted
        is_sorted = True
        for i in range(first, last):
            if P[i] > P[i+1]:
                is_sorted = False
                break
        if is_sorted:
            results.append(1)
        else:
            results.append(2)
    return results

# Read input
import sys
input = sys.stdin.read
data = input().split()
idx = 0
T = int(data[idx])
idx += 1
test_cases = []
for _ in range(T):
    N = int(data[idx])
    idx += 1
    P = list(map(int, data[idx:idx+N]))
    idx += N
    test_cases.append((N, P))

# Compute results
results = min_operations(T, test_cases)

# Output results
for res in results:
    print(res)