n = int(input())

divisors = [j for j in range(1, 10) if n % j == 0]

result = []
for i in range(n + 1):
    found = False
    for j in divisors:
        k = n // j
        if i % k == 0:
            result.append(str(j))
            found = True
            break
    if not found:
        result.append('-')

print(''.join(result))