# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))
p1 = p[0]
p_rest = p[1:]
max_p_rest = 0
for pi in p_rest:
    if pi > max_p_rest:
        max_p_rest = pi

ans = max(0, max_p_rest - p1 + 1)
print(ans)