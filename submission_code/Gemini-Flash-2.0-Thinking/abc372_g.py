def solve():
    N = int(input())
    ABC = []
    for _ in range(N):
        a, b, c = map(int, input().split())
        ABC.append((a, b, c))

    count = 0
    for x in range(1, 2001):  # Iterate up to a reasonable limit
        for y in range(1, 2001):  # Iterate up to a reasonable limit
            satisfies_all = True
            for a, b, c in ABC:
                if a * x + b * y >= c:
                    satisfies_all = False
                    break
            if satisfies_all:
                count += 1
    return count

T = int(input())
for _ in range(T):
    result = solve()
    print(result)