# YOUR CODE HERE
N = int(input())
steps = list(map(int, input().split()))

weekly_sums = []
for i in range(N):
    start_idx = i * 7
    end_idx = start_idx + 7
    weekly_sum = sum(steps[start_idx:end_idx])
    weekly_sums.append(weekly_sum)

print(' '.join(map(str, weekly_sums)))