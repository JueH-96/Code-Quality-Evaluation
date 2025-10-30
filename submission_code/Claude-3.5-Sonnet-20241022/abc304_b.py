N = int(input())

if N <= 999:
    print(N)
elif N <= 9999:
    print(N // 10 * 10)
elif N <= 99999:
    print(N // 100 * 100)
elif N <= 999999:
    print(N // 1000 * 1000)
elif N <= 9999999:
    print(N // 10000 * 10000)
elif N <= 99999999:
    print(N // 100000 * 100000)
else:
    print(N // 1000000 * 1000000)