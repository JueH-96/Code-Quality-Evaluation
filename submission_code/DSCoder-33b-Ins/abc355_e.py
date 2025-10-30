# YOUR CODE HERE
N, L, R = map(int, input().split())

def get_remainder(N, L, R):
    total = 0
    for i in range(N):
        for j in range(2**i):
            l = 2**i * j
            r = 2**i * (j+1) - 1
            if r < L or R < l:
                continue
            print("?", i, j)
            T = int(input())
            if T == -1:
                return
            total += T
    print("!", total % 100)

get_remainder(N, L, R)