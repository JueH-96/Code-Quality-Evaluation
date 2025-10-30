X, Y, U, V = map(int, input().split())

ans1 = (U - X) // 2 * (V - Y) // 2
ans2 = (U - X + 1) // 2 * (V - Y + 1) // 2
if X % 2 == 0:
    ans2 += 1
if Y % 2 == 0:
    ans2 += 1
if X % 2 == 0 and Y % 2 == 0:
    ans2 -= 1

ans = (ans1 + ans2) * 2

print(ans)