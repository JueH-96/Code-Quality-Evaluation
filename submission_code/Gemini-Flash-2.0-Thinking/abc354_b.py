def solve():
    n = int(input())
    users = []
    for _ in range(n):
        s, c = input().split()
        users.append((s, int(c)))

    users.sort()

    total_rating = sum(c for _, c in users)
    winner_index = total_rating % n

    print(users[winner_index][0])

# YOUR CODE HERE
solve()