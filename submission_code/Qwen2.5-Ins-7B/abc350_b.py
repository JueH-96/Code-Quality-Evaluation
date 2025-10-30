# YOUR CODE HERE
n, q = map(int, input().split())
holes = [1] * n
for t in map(int, input().split()):
    if holes[t-1]:
        holes[t-1] = 0
    else:
        holes[t-1] = 1
print(sum(holes))