N = int(input())

if N % 2 == 1:
    # Odd length: one '=' in the center
    result = ['-'] * N
    result[N // 2] = '='
else:
    # Even length: two '=' in the center
    result = ['-'] * N
    result[N // 2 - 1] = '='
    result[N // 2] = '='

print(''.join(result))