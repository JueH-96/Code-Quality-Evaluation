# Read the input for Team Takahashi's scores in the top of each inning
A = list(map(int, input().split()))
# Read the input for Team Aoki's scores in the bottom of each inning (up to the 8th)
B = list(map(int, input().split()))

# Calculate the total runs for Team Takahashi
total_A = sum(A)
# Calculate the total runs for Team Aoki up to the 8th inning
total_B = sum(B)

# Determine the minimum runs needed in the 9th inning for Team Aoki to win
# They need to have total_B + x > total_A
# So x > total_A - total_B
# Since x must be an integer, x >= (total_A - total_B) + 1
required_runs = (total_A - total_B) + 1

# Ensure that required_runs is at least 0 (though the constraints ensure it's >= 0)
required_runs = max(required_runs, 0)

print(required_runs)