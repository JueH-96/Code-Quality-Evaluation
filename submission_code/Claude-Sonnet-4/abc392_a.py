# YOUR CODE HERE
A1, A2, A3 = map(int, input().split())

# Check all possible permutations
if (A1 * A2 == A3 or 
    A1 * A3 == A2 or 
    A2 * A1 == A3 or 
    A2 * A3 == A1 or 
    A3 * A1 == A2 or 
    A3 * A2 == A1):
    print("Yes")
else:
    print("No")