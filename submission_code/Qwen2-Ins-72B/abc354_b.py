N = int(input())
users = [input().split() for _ in range(N)]
users.sort()
total_c = sum(int(c) for _, c in users)
winner_index = total_c % N
print(users[winner_index][0])