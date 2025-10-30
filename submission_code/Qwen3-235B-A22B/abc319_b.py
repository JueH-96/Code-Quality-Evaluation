n = int(input())
valid_js = [j for j in range(1, 10) if n % j == 0]
valid_js.sort()
result = []
for i in range(n + 1):
    for j in valid_js:
        divisor = n // j
        if divisor == 0:
            continue  # This case shouldn't happen with valid_js
        if i % divisor == 0:
            result.append(str(j))
            break
    else:
        result.append('-')
print(''.join(result))