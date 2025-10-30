# YOUR CODE HERE
B = int(input())

# Start checking from A = 1
A = 1
while True:
    # Calculate A^A
    power = A ** A
    
    # If we found a match
    if power == B:
        print(A)
        break
    
    # If A^A exceeds B, no solution exists
    if power > B:
        print(-1)
        break
    
    # Try next value of A
    A += 1