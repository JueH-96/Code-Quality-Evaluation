import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + N]))
    index += N

    # Check if the sequence can be made non-decreasing
    if all(A[i] <= A[i + 1] for i in range(N - 1)):
        results.append("Yes")
    else:
        results.append("No")

for result in results:
    print(result)