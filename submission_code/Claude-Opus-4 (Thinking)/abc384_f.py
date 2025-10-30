# YOUR CODE HERE
def f(x):
    while x % 2 == 0:
        x //= 2
    return x

N = int(input())
A = list(map(int, input().split()))

# Count occurrences of each sum
sum_count = {}
for i in range(N):
    for j in range(i, N):
        s = A[i] + A[j]
        sum_count[s] = sum_count.get(s, 0) + 1

# Compute result
result = 0
for s, count in sum_count.items():
    result += f(s) * count

print(result)