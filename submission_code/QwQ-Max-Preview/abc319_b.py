n = int(input())
valid_js = [j for j in range(1, 10) if n % j == 0]
result = []
for i in range(n + 1):
    s = '-'
    for j in valid_js:
        d = n // j
        if i % d == 0:
            s = str(j)
            break
    result.append(s)
print(''.join(result))