import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
bases = []

index = 1
for i in range(N):
    W_i = int(data[index])
    X_i = int(data[index + 1])
    bases.append((W_i, X_i))
    index += 2

def can_participate(X_i, meeting_start_utc):
    local_start = (meeting_start_utc + X_i) % 24
    local_end = (meeting_start_utc + 1 + X_i) % 24
    if local_end < local_start:
        local_end += 24
    return 9 <= local_start < 18 and 9 <= local_end - 1 < 18

max_participants = 0

for meeting_start_utc in range(24):
    participants = 0
    for W_i, X_i in bases:
        if can_participate(X_i, meeting_start_utc):
            participants += W_i
    max_participants = max(max_participants, participants)

print(max_participants)