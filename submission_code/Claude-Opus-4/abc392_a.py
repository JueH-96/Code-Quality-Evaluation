# YOUR CODE HERE
A1, A2, A3 = map(int, input().split())

# Check all possible permutations
# We need to check if any two numbers multiply to give the third

# Check if A1 * A2 = A3
if A1 * A2 == A3:
    print("Yes")
# Check if A1 * A3 = A2
elif A1 * A3 == A2:
    print("Yes")
# Check if A2 * A3 = A1
elif A2 * A3 == A1:
    print("Yes")
else:
    print("No")