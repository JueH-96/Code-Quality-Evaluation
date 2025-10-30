# YOUR CODE HERE
N = int(input())
steps = list(map(int, input().split()))

weekly_sums = []
for week in range(N):
    start_idx = week * 7
    end_idx = start_idx + 7
    week_sum = sum(steps[start_idx:end_idx])
    weekly_sums.append(week_sum)

print(' '.join(map(str, weekly_sums)))