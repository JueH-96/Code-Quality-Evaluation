def solve():
    n = int(input())
    users = []
    for _ in range(n):
        s, c = input().split()
        users.append((s, int(c)))
    
    users.sort(key=lambda x: x[0])
    
    total_rating = sum(c for _, c in users)
    winner_index = total_rating % n
    
    print(users[winner_index][0])

solve()