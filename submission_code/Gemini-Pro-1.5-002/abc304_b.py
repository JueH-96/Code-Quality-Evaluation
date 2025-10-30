# YOUR CODE HERE
n = int(input())

if n < 1000:
    print(n)
elif n < 10000:
    print(n // 10 * 10)
elif n < 100000:
    print(n // 100 * 100)
elif n < 1000000:
    print(n // 1000 * 1000)
elif n < 10000000:
    print(n // 10000 * 10000)
elif n < 100000000:
    print(n // 100000 * 100000)
else:
    print(n // 1000000 * 1000000)