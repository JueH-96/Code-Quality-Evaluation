N = int(input())

if N <= 999:
    result = N
elif N <= 9999:
    result = (N // 10) * 10
elif N <= 99999:
    result = (N // 100) * 100
elif N <= 999999:
    result = (N // 1000) * 1000
elif N <= 9999999:
    result = (N // 10000) * 10000
elif N <= 99999999:
    result = (N // 100000) * 100000
else:
    result = (N // 1000000) * 1000000

print(result)