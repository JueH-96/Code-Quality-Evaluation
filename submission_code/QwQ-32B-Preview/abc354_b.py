import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    users = []
    index = 1
    for _ in range(N):
        S = data[index]
        C = int(data[index + 1])
        users.append((S, C))
        index += 2
    # Sort users based on usernames
    sorted_users = sorted(users, key=lambda x: x[0])
    # Calculate T
    T = sum(C for S, C in sorted_users)
    # Calculate T % N
    winner_number = T % N
    # Find the user with assigned number winner_number
    winner = sorted_users[winner_number][0]
    print(winner)

if __name__ == '__main__':
    main()