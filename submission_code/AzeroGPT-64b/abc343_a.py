a, b = map(int, input().split())
ans = a + b
for i in range(10):
    if i != ans:
        print(i)
        break