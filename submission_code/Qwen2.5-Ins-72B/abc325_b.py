# YOUR CODE HERE
N = int(input())
bases = []
for _ in range(N):
    W, X = map(int, input().split())
    bases.append((W, X))

max_participants = 0

for utc_hour in range(24):
    participants = 0
    for W, X in bases:
        local_start = (utc_hour + X) % 24
        local_end = (local_start + 1) % 24
        if 9 <= local_start < 18 and 9 <= local_end < 18:
            participants += W
        elif local_start <= 8 and local_end >= 9:
            participants += W
        elif local_start <= 17 and local_end >= 18:
            participants += W
    max_participants = max(max_participants, participants)

print(max_participants)