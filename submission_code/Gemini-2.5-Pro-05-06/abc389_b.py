# Read the input integer X
X = int(input())

# Initialize n to 0 and current_factorial to 1 (representing 0!)
n = 0
current_factorial = 1

# Loop while the current factorial is less than X
# In each iteration, we calculate (n+1)!
while current_factorial < X:
    n += 1  # Move to the next number
    current_factorial *= n  # Update factorial: new_factorial = old_factorial * n

# When the loop exits, current_factorial is equal to X (due to problem guarantee),
# and n is the integer N such that N! = X.
print(n)