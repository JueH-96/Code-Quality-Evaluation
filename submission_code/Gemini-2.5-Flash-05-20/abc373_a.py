# YOUR CODE HERE
count = 0
# Loop from i = 1 to 12, as per the problem statement (S_1 to S_12)
for i in range(1, 13):
    # Read the string corresponding to the current S_i
    # The input() function reads one line from standard input.
    current_s = input()
    
    # Check if the length of the string is equal to the current index i
    if len(current_s) == i:
        # If the condition is met, increment the counter
        count += 1

# After checking all 12 strings, print the final count
print(count)