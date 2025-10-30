# YOUR CODE HERE
import sys

N = int(sys.stdin.read().strip())

if N <= 10**3 - 1:
    print(N)
elif N <= 10**4 - 1:
    print(N - N % 10)
elif N <= 10**5 - 1:
    print(N - N % 100)
elif N <= 10**6 - 1:
    print(N - N % 1000)
elif N <= 10**7 - 1:
    print(N - N % 10000)
elif N <= 10**8 - 1:
    print(N - N % 100000)