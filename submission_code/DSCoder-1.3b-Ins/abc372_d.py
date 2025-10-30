# YOUR CODE HERE

def solve():
    N = int(input())
    H = list(map(int, input().split()))
    count = [0] * (N + 1)

    for i in range(N):
        for j in range(i + 1, N + 1):
            if H[i] <= H[j]:
                count[i] += 1

    print(' '.join(map(str, count[:N])))

solve()