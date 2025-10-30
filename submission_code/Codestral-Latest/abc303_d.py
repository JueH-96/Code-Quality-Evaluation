import sys
input = sys.stdin.read
data = input().split()

X = int(data[0])
Y = int(data[1])
Z = int(data[2])
S = data[3]

# Initialize variables
caps_lock_on = False
time_spent = 0

# Iterate through each character in the string S
for char in S:
    if caps_lock_on:
        if char == 'a':
            time_spent += X
        else:  # char == 'A'
            time_spent += Z
            caps_lock_on = False
    else:
        if char == 'a':
            time_spent += Z
            caps_lock_on = True
        else:  # char == 'A'
            time_spent += X

# Print the total time spent
print(time_spent)