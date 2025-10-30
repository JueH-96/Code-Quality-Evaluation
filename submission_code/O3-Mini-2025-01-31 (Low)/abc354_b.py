def main():
    import sys
    input_data = sys.stdin.read().strip().splitlines()
    if not input_data:
        return
    N = int(input_data[0])
    users = []
    total_rating = 0
    for line in input_data[1:]:
        if line.strip() == "":
            continue
        name, rating = line.split()
        rating = int(rating)
        users.append((name, rating))
        total_rating += rating

    # Sort users by their username lexicographically.
    users.sort(key=lambda x: x[0])
    # The winner is the user at position T % N
    winner_index = total_rating % N
    print(users[winner_index][0])

if __name__ == '__main__':
    main()