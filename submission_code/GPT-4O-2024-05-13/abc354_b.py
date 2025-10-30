# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    users = []
    total_rating = 0
    
    for i in range(1, N + 1):
        name, rating = data[i].split()
        rating = int(rating)
        users.append((name, rating))
        total_rating += rating
    
    users.sort()  # Sort users by their names lexicographically
    
    winner_index = total_rating % N
    winner_name = users[winner_index][0]
    
    print(winner_name)

if __name__ == "__main__":
    main()