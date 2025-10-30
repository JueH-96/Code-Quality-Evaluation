def solve():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])

    users = []
    idx = 1
    total = 0
    
    for _ in range(N):
        name = input_data[idx]
        rating = int(input_data[idx + 1])
        idx += 2
        total += rating
        users.append((name, rating))
    
    # Sort by username lexicographically
    users.sort(key=lambda x: x[0])

    # The winner is at index (total mod N)
    winner_index = total % N
    print(users[winner_index][0])

def main():
    solve()

if __name__ == "__main__":
    main()