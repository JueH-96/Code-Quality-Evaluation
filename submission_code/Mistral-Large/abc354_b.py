import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    users = []

    for i in range(N):
        name = data[2 * i + 1]
        rating = int(data[2 * i + 2])
        users.append((name, rating))

    users.sort(key=lambda x: x[0])

    total_rating = sum(user[1] for user in users)
    winner_index = total_rating % N
    winner_name = users[winner_index][0]

    print(winner_name)

if __name__ == "__main__":
    main()