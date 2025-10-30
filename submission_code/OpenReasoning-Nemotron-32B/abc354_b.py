n = int(input().strip())
users = []
for _ in range(n):
	data = input().split()
	username = data[0]
	rating = int(data[1])
	users.append((username, rating))

sorted_users = sorted(users, key=lambda x: x[0])
total_rating = sum(rating for _, rating in users)
index = total_rating % n
print(sorted_users[index][0])