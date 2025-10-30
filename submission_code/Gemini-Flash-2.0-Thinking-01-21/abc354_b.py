def solve():
    n = int(input())
    users = []
    for _ in range(n):
        s, c = input().split()
        users.append({'username': s, 'rating': int(c)})
    
    total_rating = 0
    for user in users:
        total_rating += user['rating']
        
    users.sort(key=lambda user: user['username'])
    
    winner_index = total_rating % n
    winner_username = users[winner_index]['username']
    
    print(winner_username)

if __name__ == '__main__':
    solve()