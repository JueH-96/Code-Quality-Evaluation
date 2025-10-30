# YOUR CODE HERE
A, B = map(int, input().split())

# The difference between A and B
diff = abs(A - B)

# If the difference is even, there are 3 possible values for x
if diff % 2 == 0:
    print(3)
# If the difference is odd, there are 2 possible values for x
else:
    print(2)