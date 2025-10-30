n = int(input())
rem = n % 5
if rem <= 2:
    print((n // 5) * 5)
else:
    print((n // 5 + 1) * 5)