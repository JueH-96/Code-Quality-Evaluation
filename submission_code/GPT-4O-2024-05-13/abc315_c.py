# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
flavors = {}
max_satisfaction = 0

# Read the input and organize by flavor
for i in range(N):
    F = int(data[2 * i + 1])
    S = int(data[2 * i + 2])
    if F not in flavors:
        flavors[F] = []
    heapq.heappush(flavors[F], -S)  # Use negative to simulate max-heap

# Find the maximum satisfaction
max_deliciousness = []

for flavor in flavors:
    if len(flavors[flavor]) >= 2:
        # Get the two most delicious cups of the same flavor
        s1 = -heapq.heappop(flavors[flavor])
        s2 = -heapq.heappop(flavors[flavor])
        satisfaction_same_flavor = s1 + s2 // 2
        max_satisfaction = max(max_satisfaction, satisfaction_same_flavor)
        heapq.heappush(max_deliciousness, -s1)
        heapq.heappush(max_deliciousness, -s2)
    elif len(flavors[flavor]) == 1:
        s1 = -heapq.heappop(flavors[flavor])
        heapq.heappush(max_deliciousness, -s1)

# Get the two most delicious cups of different flavors
if len(max_deliciousness) >= 2:
    s1 = -heapq.heappop(max_deliciousness)
    s2 = -heapq.heappop(max_deliciousness)
    satisfaction_diff_flavor = s1 + s2
    max_satisfaction = max(max_satisfaction, satisfaction_diff_flavor)

print(max_satisfaction)