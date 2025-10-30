def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    users = []
    for i in range(1, n + 1):
        parts = data[i].split()
        s = parts[0]
        c = int(parts[1])
        users.append((s, c))
    
    users_sorted = sorted(users, key=lambda x: x[0])
    total_rating = sum(c for _, c in users)
    winner_index = total_rating % n
    print(users_sorted[winner_index][0])

if __name__ == "__main__":
    main()