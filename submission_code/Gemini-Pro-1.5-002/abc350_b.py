# YOUR CODE HERE
n, q = map(int, input().split())
teeth = [1] * n
for _ in range(q):
    t = int(input()) - 1
    teeth[t] = 1 - teeth[t]
print(sum(teeth))