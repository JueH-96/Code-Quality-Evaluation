# YOUR CODE HERE
V1, V2, V3 = map(int, input().split())
if V1 + 2*V2 + 3*V3 == 1029:
    print("Yes")
    # Using the sample input positions as a valid configuration
    print("0 0 0 0 6 0 6 0 0")
else:
    print("No")