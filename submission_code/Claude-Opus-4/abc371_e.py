# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

total = 0
for i in range(n):
    seen = set()
    for j in range(i, n):
        seen.add(a[j])
        total += len(seen)

print(total)