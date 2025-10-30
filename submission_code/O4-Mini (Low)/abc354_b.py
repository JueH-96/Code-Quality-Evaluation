def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    users = []
    total_rating = 0
    for _ in range(N):
        s, c = input().split()
        c = int(c)
        users.append((s, c))
        total_rating += c

    # Sort usernames lexicographically
    users.sort(key=lambda x: x[0])
    
    # Determine winner index
    winner_index = total_rating % N
    # Print the corresponding username
    print(users[winner_index][0])

if __name__ == "__main__":
    main()