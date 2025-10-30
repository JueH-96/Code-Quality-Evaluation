def calculate_min_time(X, Y, Z, S):
    time_spent = 0
    caps_lock = False

    for char in S:
        if char.isupper():
            if caps_lock:
                time_spent += X
            else:
                # Check if it's cheaper to use Shift or to toggle Caps Lock and then type
                if Y < Z + X:
                    time_spent += Y
                else:
                    time_spent += Z + X
                    caps_lock = True
        else:
            if caps_lock:
                # Check if it's cheaper to use Shift or to toggle Caps Lock and then type
                if Y < Z + X:
                    time_spent += Y
                else:
                    time_spent += Z + X
                    caps_lock = False
            else:
                time_spent += X

    return time_spent

# Read input
X, Y, Z = map(int, input().split())
S = input()

# Calculate and print the answer
print(calculate_min_time(X, Y, Z, S))