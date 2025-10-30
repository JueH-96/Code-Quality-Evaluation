N = int(input())
valid_js = [j for j in range(1, 10) if N % j == 0]
valid_js.sort()
result = []
for i in range(N + 1):
    for j in valid_js:
        d_j = N // j
        if i % d_j == 0:
            result.append(str(j))
            break
    else:
        result.append('-')
print(''.join(result))