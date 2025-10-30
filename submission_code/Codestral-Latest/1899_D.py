import sys
from collections import defaultdict

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

    count_dict = defaultdict(int)
    for num in a:
        count_dict[num] += 1

    total_pairs = 0
    for count in count_dict.values():
        if count > 1:
            total_pairs += count * (count - 1) // 2

    results.append(total_pairs)

for result in results:
    print(result)