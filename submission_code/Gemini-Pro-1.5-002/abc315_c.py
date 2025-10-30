# YOUR CODE HERE
def solve():
    n = int(input())
    cups = []
    for _ in range(n):
        f, s = map(int, input().split())
        cups.append((f, s))

    max_satisfaction = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            f1, s1 = cups[i]
            f2, s2 = cups[j]
            
            satisfaction = 0
            if f1 != f2:
                satisfaction = max(s1, s2) + min(s1, s2)
            else:
                satisfaction = max(s1, s2) + min(s1, s2) // 2
            
            max_satisfaction = max(max_satisfaction, satisfaction)
            
    print(max_satisfaction)

solve()