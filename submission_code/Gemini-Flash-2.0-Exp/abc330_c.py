D = int(input())
ans = float('inf')
x = 0
while x * x <= D:
    y = int((D - x * x)**0.5)
    ans = min(ans, abs(x * x + y * y - D))
    y += 1
    ans = min(ans, abs(x * x + y * y - D))
    x += 1
print(ans)