n = int(input())
ans = ["o"] * n
for i in range(3, n + 1, 3):
    ans[i - 1] = "x"
print("".join(ans))