n, m = map(int, input().split())
votes = list(map(int, input().split()))
count = [0] * (n + 1)
max_count = 0
current_leader = None

for a in votes:
    count[a] += 1
    new_count = count[a]
    if new_count > max_count:
        max_count = new_count
        current_leader = a
    elif new_count == max_count:
        if a < current_leader:
            current_leader = a
    print(current_leader)