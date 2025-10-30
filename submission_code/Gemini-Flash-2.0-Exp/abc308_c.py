def solve():
    n = int(input())
    people = []
    for i in range(n):
        a, b = map(int, input().split())
        people.append((a, b, i + 1))

    success_rates = []
    for a, b, i in people:
        success_rates.append((a / (a + b), i))

    success_rates.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    result = [str(x[1]) for x in success_rates]
    print(" ".join(result))

solve()