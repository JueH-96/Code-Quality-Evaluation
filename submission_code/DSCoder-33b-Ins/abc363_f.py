def solve(N):
    if N < 10:
        return str(N)
    for i in range(2, 10**6 + 1):
        if N % i == 0:
            s = str(i) + '*' + str(N // i)
            if s == s[::-1]:
                return s
    return -1

N = int(input())
print(solve(N))