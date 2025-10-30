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
            
            if s1 >= s2:
                s = s1
                t = s2
            else:
                s = s2
                t = s1

            if f1 != f2:
                satisfaction = s + t
            else:
                satisfaction = s + t // 2
            
            max_satisfaction = max(max_satisfaction, satisfaction)
    
    print(max_satisfaction)

solve()