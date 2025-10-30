import math

def solve():
    n = int(input())
    cups = []
    for _ in range(n):
        f, s = map(int, input().split())
        cups.append({'flavor': f, 'deliciousness': s})
    
    max_satisfaction = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            cup1 = cups[i]
            cup2 = cups[j]
            s1 = cup1['deliciousness']
            s2 = cup2['deliciousness']
            f1 = cup1['flavor']
            f2 = cup2['flavor']
            
            s = max(s1, s2)
            t = min(s1, s2)
            
            current_satisfaction = 0
            if f1 == f2:
                current_satisfaction = s + t / 2
            else:
                current_satisfaction = s + t
                
            max_satisfaction = max(max_satisfaction, current_satisfaction)
            
    print(int(max_satisfaction))

if __name__ == '__main__':
    solve()