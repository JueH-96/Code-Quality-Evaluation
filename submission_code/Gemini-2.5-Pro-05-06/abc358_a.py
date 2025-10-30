# YOUR CODE HERE
# Read the input strings S and T
s_t_input = input().split()
S = s_t_input[0]
T = s_t_input[1]

# Define the target strings
target_S = "AtCoder"
target_T = "Land"

# Check if S matches target_S and T matches target_T
if S == target_S and T == target_T:
    print("Yes")
else:
    print("No")