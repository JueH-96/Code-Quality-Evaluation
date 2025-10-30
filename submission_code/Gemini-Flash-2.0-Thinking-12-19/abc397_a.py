import sys

# Read the input temperature X as a float
X = float(sys.stdin.readline())

# Classify the temperature and print the corresponding integer
if X >= 38.0:
    print(1)
elif X >= 37.5:
    print(2)
else:
    print(3)