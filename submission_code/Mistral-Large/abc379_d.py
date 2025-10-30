import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

Q = int(data[0])
index = 1

plants = defaultdict(int)
total_height_increase = 0
results = []

for _ in range(Q):
    query_type = int(data[index])

    if query_type == 1:
        plants[0] += 1
        index += 1
    elif query_type == 2:
        T = int(data[index + 1])
        total_height_increase += T
        index += 2
    elif query_type == 3:
        H = int(data[index + 1])
        harvest_count = 0

        for height in list(plants.keys()):
            if height + total_height_increase >= H:
                harvest_count += plants[height]
                del plants[height]

        results.append(harvest_count)
        index += 2

for result in results:
    print(result)