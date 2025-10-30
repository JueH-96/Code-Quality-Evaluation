# Read the input from stdin
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total score of Team Takahashi and Team Aoki
total_A = sum(A)
total_B = sum(B)

# Calculate the minimum number of runs Team Aoki needs to score in the bottom of the ninth to win
min_runs = total_A - total_B + 1

print(min_runs)