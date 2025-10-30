N = int(input())

# Find divisors of N between 1 and 9
divisors = []
for j in range(1, 10):
    if N % j == 0:
        divisors.append(j)

result = []
for i in range(N + 1):
    found = False
    for j in divisors:
        if i % (N // j) == 0:
            result.append(str(j))
            found = True
            break
    if not found:
        result.append('-')

print(''.join(result))