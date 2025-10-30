# YOUR CODE HERE
m, d = map(int, input().split())
y, mo, da = map(int, input().split())

if da == d:
    da = 1
    if mo == m:
        mo = 1
        y += 1
    else:
        mo += 1
else:
    da += 1

print(y, mo, da)