def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    users = []
    idx = 1
    total_rating = 0
    for _ in range(N):
        name = input_data[idx]
        rating = int(input_data[idx + 1])
        idx += 2
        users.append((name, rating))
        total_rating += rating
    
    # Sort users by name in lexicographical order
    users.sort(key=lambda x: x[0])
    
    # The winner's index based on (sum of ratings) mod N
    winner_index = total_rating % N
    
    # Print the username of the winner
    print(users[winner_index][0])

# Do not forget to call main()
if __name__ == "__main__":
    main()