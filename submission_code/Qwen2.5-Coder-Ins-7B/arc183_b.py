import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1

results = []

for _ in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    A = list(map(int, data[index + 2:index + 2 + N]))
    B = list(map(int, data[index + 2 + N:index + 2 + 2 * N]))
    index += 2 + 2 * N

    # Check if it's possible to make A identical to B
    if all(A.count(x) == B.count(x) for x in set(A)):
        results.append("Yes")
    else:
        results.append("No")

# Print the results
for result in results:
    print(result)