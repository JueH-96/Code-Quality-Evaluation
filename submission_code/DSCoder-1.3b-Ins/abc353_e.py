def f(x, y):
    i = 0
    while i < len(x) and i < len(y):
        if x[i] != y[i]:
            break
        i += 1
    return i

def solve():
    N = int(input())
    S = [input() for _ in range(N)]
    result = 0
    for i in range(N):
        for j in range(i+1, N):
            result += f(S[i], S[j])
    print(result)

solve()