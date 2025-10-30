bits = list(map(int, input().split()))
result = 0
for i in range(64):
    result += bits[i] * (2 ** i)
print(result)