N = int(input())
divisors = []
for j in range(1, 10):
    if N % j == 0:
        divisors.append(j)
divisors.sort()  # Ensure we check in order from smallest to largest

result = []
for i in range(N + 1):
    current = '-'
    for j in divisors:
        d = N // j
        if i % d == 0:
            current = str(j)
            break
    result.append(current)

print(''.join(result))