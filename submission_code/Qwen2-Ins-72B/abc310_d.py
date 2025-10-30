from itertools import product

N, T, M = map(int, input().split())
AB = [tuple(map(int, input().split())) for _ in range(M)]

def solve():
    if M == 0:
        return T**N - T

    ans = 0
    for comb in product(range(1, T+1), repeat=N):
        for a, b in AB:
            if comb[a-1] == comb[b-1]:
                break
        else:
            ans += 1
    return ans

print(solve())