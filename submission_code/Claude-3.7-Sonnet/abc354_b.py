def solve():
    # Read the number of users
    N = int(input())
    
    # Read each user's name and rating
    users = []
    total_rating = 0
    
    for _ in range(N):
        name, rating = input().split()
        rating = int(rating)
        users.append((name, rating))
        total_rating += rating
    
    # Sort users by name lexicographically
    users.sort(key=lambda x: x[0])
    
    # Find the winner based on the sum modulo N
    winner_index = total_rating % N
    
    # Print the winner's name
    print(users[winner_index][0])

# Execute the solution
solve()