def solve():
    n = int(input())
    cups = []
    for _ in range(n):
        f, s = map(int, input().split())
        cups.append((f, s))

    max_satisfaction = 0
    for i in range(n):
        for j in range(i + 1, n):
            flavor1, deliciousness1 = cups[i]
            flavor2, deliciousness2 = cups[j]

            s = max(deliciousness1, deliciousness2)
            t = min(deliciousness1, deliciousness2)

            if flavor1 != flavor2:
                current_satisfaction = s + t
            else:
                current_satisfaction = s + t // 2

            max_satisfaction = max(max_satisfaction, current_satisfaction)

    print(max_satisfaction)

solve()