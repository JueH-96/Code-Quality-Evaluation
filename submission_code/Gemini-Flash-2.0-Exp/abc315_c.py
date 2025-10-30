def solve():
    n = int(input())
    cups = []
    for _ in range(n):
        f, s = map(int, input().split())
        cups.append((f, s))

    max_satisfaction = 0
    for i in range(n):
        for j in range(i + 1, n):
            f1, s1 = cups[i]
            f2, s2 = cups[j]
            
            if f1 != f2:
                max_satisfaction = max(max_satisfaction, s1 + s2)
            else:
                s = max(s1, s2)
                t = min(s1, s2)
                max_satisfaction = max(max_satisfaction, s + t / 2)

    print(int(max_satisfaction))

solve()