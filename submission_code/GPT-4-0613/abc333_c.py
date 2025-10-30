N = int(input().strip())
if N <= 3:
    print(N)
elif N <= 11:
    print(10 + N)
else:
    print(110 * (N - 11) + 33)