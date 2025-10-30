import sys
data = sys.stdin.read().split()
X = int(data[0])
Y = int(data[1])
Z = int(data[2])
S = data[3]

# Initialize dp for i=0: cost with caps off and on with no characters typed
prev_off = 0
prev_on = Z

# Iterate through each character in the string S
for char in S:
    if char == 'a':
        # Cost to append 'a' with caps off and on
        cost_off = X
        cost_on = Y
    else:  # char == 'A'
        # Cost to append 'A' with caps off and on
        cost_off = Y
        cost_on = X
    
    # Compute the new dp values for current character
    curr_off = min(prev_off + cost_off, prev_on + Z + cost_off)
    curr_on = min(prev_on + cost_on, prev_off + Z + cost_on)
    
    # Update previous dp values for next iteration
    prev_off = curr_off
    prev_on = curr_on

# The minimum time is the minimum of the final dp values
answer = min(prev_off, prev_on)
print(answer)