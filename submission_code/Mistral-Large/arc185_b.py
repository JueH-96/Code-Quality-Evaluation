import sys
input = sys.stdin.read
data = input().split()

index = 0
T = int(data[index])
index += 1
results = []

for _ in range(T):
    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + N]))
    index += N

    # Calculate the cumulative sum and the expected cumulative sum if the sequence were non-decreasing
    cumulative_sum = [0] * (N + 1)
    expected_sum = [0] * (N + 1)

    for i in range(N):
        cumulative_sum[i + 1] = cumulative_sum[i] + A[i]
        expected_sum[i + 1] = expected_sum[i] + A[i] // (i + 1) * (i + 1)

    # Check if the sequence can be made non-decreasing
    possible = True
    for i in range(1, N + 1):
        if cumulative_sum[i] < expected_sum[i]:
            possible = False
            break

    results.append("Yes" if possible else "No")

sys.stdout.write("
".join(results) + "
")