N = int(input())
result = ''.join(['1' if i % 2 == 0 else '0' for i in range(2 * N + 1)])
print(result)