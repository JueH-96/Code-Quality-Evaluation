# Read the input
N = int(input())
S = input()

# Initialize the sum
total_sum = 0

# Iterate through all possible substrings
for i in range(1, N+1):
    for j in range(i, N+1):
        # Convert the substring to an integer and add it to the sum
        total_sum += int(S[i-1:j])

# Print the answer
print(total_sum)