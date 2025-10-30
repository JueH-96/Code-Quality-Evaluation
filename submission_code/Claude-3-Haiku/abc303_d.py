# Read the input
X, Y, Z = map(int, input().split())
S = input()

# Initialize the state
caps_lock = False
time = 0
result = ""

# Iterate through the string S
for c in S:
    if c == 'A':
        if caps_lock:
            time += X
            result += 'A'
        else:
            time += Y
            result += 'A'
    else:
        if caps_lock:
            time += Y
            result += 'a'
        else:
            time += X
            result += 'a'
    
    # Toggle the Caps Lock state
    time += Z
    caps_lock = not caps_lock

print(time)