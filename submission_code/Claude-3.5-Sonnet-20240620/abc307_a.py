# YOUR CODE HERE
N = int(input())
steps = list(map(int, input().split()))

weekly_totals = []
for i in range(N):
    start = i * 7
    end = start + 7
    weekly_total = sum(steps[start:end])
    weekly_totals.append(weekly_total)

print(*weekly_totals)