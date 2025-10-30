N = int(input())
divs = [j for j in range(1, 10) if N % j == 0]
result = []
for i in range(N + 1):
    current = '-'
    for j in divs:
        q = N // j
        if i % q == 0:
            current = str(j)
            break
    result.append(current)
print(''.join(result))