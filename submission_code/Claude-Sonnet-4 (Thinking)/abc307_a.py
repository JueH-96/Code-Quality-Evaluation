n = int(input())
steps = list(map(int, input().split()))

weekly_totals = []
for week in range(n):
    start = week * 7
    end = start + 7
    weekly_total = sum(steps[start:end])
    weekly_totals.append(weekly_total)

print(' '.join(map(str, weekly_totals)))