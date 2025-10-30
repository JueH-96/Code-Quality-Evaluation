# YOUR CODE HERE

N = int(input())

if N < 10**3:
    print(N)
elif N < 10**4:
    print(N // 10 * 10)
elif N < 10**5:
    print(N // 100 * 100)
elif N < 10**6:
    print(N // 1000 * 1000)
elif N < 10**7:
    print(N // 10000 * 10000)
elif N < 10**8:
    print(N // 100000 * 100000)
elif N < 10**9:
    print(N // 1000000 * 1000000)