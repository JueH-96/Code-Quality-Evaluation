n = int(input())
divisors = [j for j in range(1, 10) if n % j == 0]
divisors.sort()

result = []
for i in range(n + 1):
    char = '-'
    for j in divisors:
        if i % (n // j) == 0:
            char = str(j)
            break
    result.append(char)
print(''.join(result))