def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    N = int(input_data[0])
    total_rating = 0
    users = []
    
    for i in range(1, N+1):
        parts = input_data[i].split()
        username = parts[0]
        rating = int(parts[1])
        users.append(username)
        total_rating += rating
    
    # Sort the usernames in lexicographical order
    users.sort()
    
    # Determine the winning index and print the username.
    winner_index = total_rating % N
    print(users[winner_index])
    
if __name__ == '__main__':
    main()