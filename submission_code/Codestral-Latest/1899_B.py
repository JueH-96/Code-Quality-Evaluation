import sys
from itertools import accumulate

input = sys.stdin.read
data = input().split()

index = 0
t = int(data[index])
index += 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n

    prefix_sum = list(accumulate(a))
    total_weight = prefix_sum[-1]

    max_diff = 0

    for k in range(1, n + 1):
        if n % k == 0:
            truck_weight = prefix_sum[k - 1]
            remaining_weight = total_weight - truck_weight
            num_trucks = n // k
            max_diff = max(max_diff, abs(remaining_weight - truck_weight * (num_trucks - 1)))

    results.append(max_diff)

for result in results:
    print(result)