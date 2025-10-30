# YOUR CODE HERE
k, g, m = map(int, input().split())
glass = 0
mug = 0
for i in range(k):
    if glass == g:
        glass = 0
    elif mug == 0:
        mug = m
    else:
        transfer = min(mug, g - glass)
        glass += transfer
        mug -= transfer
    if glass == g:
        glass = 0

print(glass, mug)