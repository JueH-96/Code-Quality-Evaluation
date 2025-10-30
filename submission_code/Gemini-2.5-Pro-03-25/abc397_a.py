import sys

def solve():
    # Read the input temperature as a float
    x = float(sys.stdin.readline())

    # Classify the temperature based on the given rules
    if x >= 38.0:
        print(1)  # High fever
    elif x >= 37.5:
        print(2)  # Fever
    else:
        print(3)  # Normal

# Call the solve function to execute the logic
solve()