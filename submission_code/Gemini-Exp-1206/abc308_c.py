def solve():
    n = int(input())
    rates = []
    for i in range(n):
        a, b = map(int, input().split())
        rates.append((a / (a + b), - (i + 1)))
    
    rates.sort(reverse=True)
    
    for rate, neg_person_num in rates:
        print(-neg_person_num, end=" ")
    print()

solve()