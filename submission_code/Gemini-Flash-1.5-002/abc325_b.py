# YOUR CODE HERE
N = int(input())
bases = []
for i in range(N):
    W, X = map(int, input().split())
    bases.append((W, X))

max_employees = 0
for meeting_start_utc in range(24):
    total_employees = 0
    for W, X in bases:
        meeting_start_local = (meeting_start_utc - X) % 24
        meeting_end_local = (meeting_start_utc - X + 1) % 24
        if meeting_start_local >= 9 and meeting_end_local <= 18:
            total_employees += W
        elif meeting_start_local < 9 and meeting_end_local <= 18 and meeting_end_local >=9:
            total_employees += W
        elif meeting_start_local >=9 and meeting_end_local > 18 and meeting_start_local < 18:
            total_employees += W

    max_employees = max(max_employees, total_employees)

print(max_employees)