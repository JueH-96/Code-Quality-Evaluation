def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    entries = []
    idx = 1
    total_rating = 0
    for _ in range(n):
        name = data[idx]
        rating = int(data[idx+1])
        idx += 2
        entries.append((name, rating))
        total_rating += rating

    # Sort by username (lexicographical order)
    entries.sort(key=lambda x: x[0])

    # Determine winner
    winner_index = total_rating % n
    print(entries[winner_index][0])

if __name__ == "__main__":
    main()