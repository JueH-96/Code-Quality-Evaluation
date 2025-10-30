# Read the input temperature
X = float(input())

# Classify the temperature and output the corresponding integer
if X >= 38.0:
    print(1)
elif X >= 37.5:
    print(2)
else:
    print(3)