# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

counts = [0] * (N + 1)
for a in A:
    counts[a] += 1

answer = sum(1 for c in counts if c == 2)
print(answer)