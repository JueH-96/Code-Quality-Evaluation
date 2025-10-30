# Read Takahashi's runs for each inning
takahashi_runs = list(map(int, input().split()))

# Read Aoki's runs for each inning
aoki_runs = list(map(int, input().split()))

# Calculate the total score for each team
takahashi_total = sum(takahashi_runs)
aoki_total = sum(aoki_runs)

# Calculate the minimum runs Aoki needs to win
# Since Aoki needs to score strictly more than Takahashi to win,
# we need takahashi_total < aoki_total + ninth_inning_runs
# So the minimum is takahashi_total - aoki_total + 1
min_runs_needed = takahashi_total - aoki_total + 1

print(min_runs_needed)