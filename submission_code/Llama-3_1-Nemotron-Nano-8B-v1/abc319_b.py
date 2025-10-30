n = int(input())
divisors = []
for j in range(1, 10):
    if n % j == 0:
        divisors.append(j)
result = []
for i in range(n + 1):
    for j in divisors:
        k = n // j
        if i % k == 0:
            result.append(str(j))
            break
    else:
        result.append('-')
print(''.join(result))