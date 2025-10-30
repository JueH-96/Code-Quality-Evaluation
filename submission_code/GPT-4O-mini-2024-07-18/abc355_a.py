# YOUR CODE HERE
def identify_culprit(A, B):
    # Create a set of all suspects
    suspects = {1, 2, 3}
    
    # Remove the suspects that are not culprits according to the witnesses
    suspects.discard(A)
    suspects.discard(B)
    
    # If there's exactly one suspect left, return that suspect
    if len(suspects) == 1:
        return suspects.pop()
    else:
        return -1

# Read input
A, B = map(int, input().split())
# Get the result
result = identify_culprit(A, B)
# Print the result
print(result)