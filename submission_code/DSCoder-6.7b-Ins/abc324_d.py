from itertools import permutations

def solve():
    N = int(input())
    S = input()
    digits = list(map(int, S))
    perms = list(permutations(range(1, N+1)))
    squares = set()
    for perm in perms:
        num = 0
        for i in range(N):
            num += digits[i] * 10 ** (N-i-1)
        sqrt = num ** 0.5
        if sqrt == int(sqrt):
            squares.add(num)
    print(len(squares))

solve()