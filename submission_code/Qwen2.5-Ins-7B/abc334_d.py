# YOUR CODE HERE
n, q = map(int, input().split())
R = list(map(int, input().split()))
queries = [int(input()) for _ in range(q)]

R.sort()
prefix_sum = [0]
for r in R:
    prefix_sum.append(prefix_sum[-1] + r)

def max_sleighs(x):
    i = 0
    for i, ps in enumerate(prefix_sum):
        if ps > x:
            break
    return i

for x in queries:
    print(max_sleighs(x))