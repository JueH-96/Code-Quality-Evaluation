# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
employees = []
for i in range(n):
    w, x = int(data[2*i+1]), int(data[2*i+2])
    employees.append((w, x))

max_participants = 0

# Check all possible meeting times from 9:00 to 18:00
for start in range(9, 18):
    participants = 0
    for w, x in employees:
        if 9 <= (start - x) % 24 <= 18:
            participants += w
    max_participants = max(max_participants, participants)

print(max_participants)