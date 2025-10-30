import sys
import bisect

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
X = list(map(int, data[index:index + N]))
index += N
P = list(map(int, data[index:index + N]))
index += N
Q = int(data[index])
index += 1
queries = []
for _ in range(Q):
    L = int(data[index])
    R = int(data[index + 1])
    queries.append((L, R))
    index += 2

# Preprocess the cumulative sum of villagers
cumulative_sum = []
current_sum = 0
for i in range(N):
    current_sum += P[i]
    cumulative_sum.append(current_sum)

def get_villagers_count(L, R):
    left_index = bisect.bisect_left(X, L)
    right_index = bisect.bisect_right(X, R) - 1

    if left_index > right_index or right_index < 0:
        return 0

    if left_index == 0:
        return cumulative_sum[right_index]
    else:
        return cumulative_sum[right_index] - cumulative_sum[left_index - 1]

results = []
for L, R in queries:
    results.append(get_villagers_count(L, R))

sys.stdout.write("
".join(map(str, results)) + "
")