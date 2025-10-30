# YOUR CODE HERE
N = int(input())
S = input().strip()

# Initialize the total sum
total_sum = 0

# Iterate over all possible substrings
for i in range(N):
    current_num = 0
    for j in range(i, N):
        # Update the current number by adding the next digit
        current_num = current_num * 10 + int(S[j])
        # Add the current number to the total sum
        total_sum += current_num

# Print the result
print(total_sum)