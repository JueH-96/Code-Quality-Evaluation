def solve():
    n = int(input())
    users = []
    for _ in range(n):
        line = input().split()
        username = line[0]
        rating = int(line[1])
        users.append({'username': username, 'rating': rating})
    
    total_rating = 0
    for user in users:
        total_rating += user['rating']
        
    sorted_users = sorted(users, key=lambda user: user['username'])
    
    winner_index = total_rating % n
    winner_username = sorted_users[winner_index]['username']
    
    print(winner_username)

if __name__ == '__main__':
    solve()