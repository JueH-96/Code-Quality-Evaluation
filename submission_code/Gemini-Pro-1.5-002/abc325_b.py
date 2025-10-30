# YOUR CODE HERE
def solve():
    n = int(input())
    bases = []
    for _ in range(n):
        w, x = map(int, input().split())
        bases.append((w, x))

    max_participants = 0
    for utc_start in range(24):
        total_participants = 0
        for w, x in bases:
            local_start = (utc_start + x) % 24
            local_end = (local_start + 1) % 24
            
            if 9 <= local_start <= 17:
                total_participants += w
            elif local_start == 18 and local_end == 9:
                total_participants +=w
            elif local_start == 8 and local_end == 9:
                continue
            elif local_start < 9 and local_end >9 and local_end <= 18:
                total_participants += w
            elif local_start < 9 and local_end > 18:
                continue

        max_participants = max(max_participants, total_participants)

    print(max_participants)

solve()