n = int(input())
result = []
for i in range(1, n + 1):
    result.append('x' if i % 3 == 0 else 'o')
print(''.join(result))