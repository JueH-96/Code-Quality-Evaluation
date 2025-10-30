# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    N = int(data[0].strip())
    users = []
    
    for i in range(1, N + 1):
        name, rating = data[i].rsplit(' ', 1)
        rating = int(rating)
        users.append((name, rating))
    
    # Sort users by their names lexicographically
    users.sort(key=lambda x: x[0])
    
    # Calculate the total sum of ratings
    total_rating = sum(rating for _, rating in users)
    
    # Determine the winner index
    winner_index = total_rating % N
    
    # Output the winner's name
    print(users[winner_index][0])

if __name__ == "__main__":
    main()