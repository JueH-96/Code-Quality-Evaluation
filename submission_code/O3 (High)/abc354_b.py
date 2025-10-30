def main():
    import sys

    # Read number of users
    n_line = sys.stdin.readline()
    while n_line.strip() == "":
        n_line = sys.stdin.readline()
    N = int(n_line.strip())

    names = []
    total_rating = 0

    # Read all (name, rating) pairs
    for _ in range(N):
        line = sys.stdin.readline().strip()
        while line == "":
            line = sys.stdin.readline().strip()
        name, rating_str = line.split()
        rating = int(rating_str)
        total_rating += rating
        names.append(name)

    # Sort names lexicographically
    names.sort()

    # Winner index
    winner_index = total_rating % N

    # Output winner's name
    print(names[winner_index])

if __name__ == "__main__":
    main()