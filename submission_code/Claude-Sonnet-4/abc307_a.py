# YOUR CODE HERE
N = int(input())
steps = list(map(int, input().split()))

weekly_sums = []
for week in range(N):
    start_index = week * 7
    end_index = start_index + 7
    week_sum = sum(steps[start_index:end_index])
    weekly_sums.append(week_sum)

print(' '.join(map(str, weekly_sums)))