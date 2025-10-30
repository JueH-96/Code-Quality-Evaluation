N = int(input())

if N % 2 == 1:
    # Odd case: single '=' in the middle
    half = (N - 1) // 2
    s = '-' * half + '=' + '-' * half
else:
    # Even case: two '=' in the middle
    half = (N - 2) // 2
    s = '-' * half + '==' + '-' * half

print(s)