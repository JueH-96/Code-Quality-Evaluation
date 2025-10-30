# YOUR CODE HERE

N, T, P = map(int, input().split())
L = list(map(int, input().split()))

def solve(N, T, P, L):
    days = 0
    while True:
        for i in range(N):
            if L[i] >= T:
                P -= 1
                if P == 0:
                    return days
        days += 1
        L = [x+1 for x in L]

print(solve(N, T, P, L))