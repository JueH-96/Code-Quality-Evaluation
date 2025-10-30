import sys

# Read the input temperature X
X = float(sys.stdin.readline())

# Classify the temperature and print the corresponding integer
if X >= 38.0:
    print(1)  # High fever
elif X >= 37.5:
    # This condition is met only if X < 38.0 (because of the previous if)
    # So this effectively checks 37.5 <= X < 38.0
    print(2)  # Fever
else:
    # This condition is met only if X < 37.5 (because of the previous elif)
    print(3)  # Normal