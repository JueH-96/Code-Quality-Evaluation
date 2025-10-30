# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    users = []
    index = 1
    for _ in range(N):
        S = data[index]
        C = int(data[index + 1])
        users.append((S, C))
        index += 2
    
    # Sort by username lexicographically
    users_sorted = sorted(users, key=lambda x: x[0])
    
    # Calculate total rating
    total = sum(c for _, c in users_sorted)
    
    # Determine the winner
    winner_index = total % N
    winner = users_sorted[winner_index][0]
    
    print(winner)

if __name__ == "__main__":
    main()