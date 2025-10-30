# YOUR CODE HERE
n = int(input())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_participants = 0

for t in range(24):
    participants = 0
    for w, x in bases:
        local_start = (t + x) % 24
        if 9 <= local_start <= 17:
            participants += w
    max_participants = max(max_participants, participants)

print(max_participants)