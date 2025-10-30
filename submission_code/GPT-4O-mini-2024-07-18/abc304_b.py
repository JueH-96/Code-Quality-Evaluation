# YOUR CODE HERE
N = int(input().strip())

if N < 1000:
    print(N)
elif N < 10000:
    print(N // 10 * 10)
elif N < 100000:
    print(N // 100 * 100)
elif N < 1000000:
    print(N // 1000 * 1000)
elif N < 10000000:
    print(N // 10000 * 10000)
elif N < 100000000:
    print(N // 100000 * 100000)