n, m = map(int, input().split())
a = list(map(int, input().split()))

counts = [0] * (n + 1)
current_max = 0
current_leader = 0

for vote in a:
    counts[vote] += 1
    if counts[vote] > current_max:
        current_max = counts[vote]
        current_leader = vote
    elif counts[vote] == current_max:
        if current_leader > vote:
            current_leader = vote
    print(current_leader)