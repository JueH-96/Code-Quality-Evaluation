def main():
    N = int(input().strip())
    users = []
    total_rating = 0
    for _ in range(N):
        data = input().split()
        name = data[0]
        rating = int(data[1])
        users.append((name, rating))
        total_rating += rating
    
    sorted_users = sorted(users, key=lambda x: x[0])
    index = total_rating % N
    winner = sorted_users[index][0]
    print(winner)

if __name__ == '__main__':
    main()