# Read the input
N = int(input())
steps = [int(x) for x in input().split()]

# Calculate the total steps for each week
week_totals = []
for i in range(N):
    week_total = sum(steps[i*7:(i+1)*7])
    week_totals.append(week_total)

# Print the results
print(" ".join(map(str, week_totals)))