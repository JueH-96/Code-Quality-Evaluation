# YOUR CODE HERE

N = int(input())

if N >= 1 and N <= 100:
    if N % 2 == 0:
        print('1' * N + '0' * (N + 1))
    else:
        print('1' * (N + 1) + '0' * N)