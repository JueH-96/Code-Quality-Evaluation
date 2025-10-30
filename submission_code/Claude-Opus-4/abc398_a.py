# YOUR CODE HERE
N = int(input())

if N % 2 == 0:
    # Even length: two '=' in the middle
    result = ['-'] * N
    mid = N // 2
    result[mid - 1] = '='
    result[mid] = '='
else:
    # Odd length: one '=' in the middle
    result = ['-'] * N
    mid = N // 2
    result[mid] = '='

print(''.join(result))