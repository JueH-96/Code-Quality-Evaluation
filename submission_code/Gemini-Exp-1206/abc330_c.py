D = int(input())
ans = D
x = 0
y = int(D**0.5) + 1
while x <= y:
    diff = x * x + y * y - D
    ans = min(ans, abs(diff))
    if diff > 0:
        y -= 1
    else:
        x += 1
print(ans)