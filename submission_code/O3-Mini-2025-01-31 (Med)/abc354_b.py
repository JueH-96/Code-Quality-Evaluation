def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # The first token is N.
    N = int(input_data[0])
    # Prepare lists and sum ratings.
    participants = []
    total_rating = 0
    idx = 1
    for _ in range(N):
        name = input_data[idx]
        rating = int(input_data[idx + 1])
        participants.append(name)
        total_rating += rating
        idx += 2
    
    # Sort the usernames in lexicographical order.
    sorted_names = sorted(participants)
    
    # Compute winning index.
    winner_index = total_rating % N
    
    # Print the winner's username.
    print(sorted_names[winner_index])

if __name__ == "__main__":
    main()