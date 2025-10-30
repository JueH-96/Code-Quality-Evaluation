import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N = int(data[0])
    users = []
    
    for i in range(1, N + 1):
        name, rating = data[i].split()
        rating = int(rating)
        users.append((name, rating))
    
    # Sort users by name lexicographically
    users.sort(key=lambda x: x[0])
    
    # Calculate the sum of ratings
    total_rating = sum(user[1] for user in users)
    
    # Find the winner index
    winner_index = total_rating % N
    
    # Print the winner's name
    print(users[winner_index][0])

if __name__ == "__main__":
    main()