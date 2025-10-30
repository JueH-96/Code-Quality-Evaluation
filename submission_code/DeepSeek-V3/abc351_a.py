# Read the input for Team Takahashi's scores in the top of each inning
A = list(map(int, input().split()))
# Read the input for Team Aoki's scores in the bottom of each inning (first 8 innings)
B = list(map(int, input().split()))

# Calculate the total runs for Team Takahashi
total_A = sum(A)
# Calculate the total runs for Team Aoki after 8 innings
total_B = sum(B)

# Determine the minimum runs needed in the bottom of the ninth for Team Aoki to win
# They need to have a total score greater than Team Takahashi
required_runs = max(0, total_A - total_B + 1)

print(required_runs)