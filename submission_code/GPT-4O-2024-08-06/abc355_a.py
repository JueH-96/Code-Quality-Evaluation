# YOUR CODE HERE
def find_culprit():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    A, B = map(int, data.split())
    
    # Create a set of all suspects
    suspects = {1, 2, 3}
    
    # Remove the suspects who are not the culprits according to witnesses
    suspects.discard(A)
    suspects.discard(B)
    
    # If there's exactly one suspect left, that's the culprit
    if len(suspects) == 1:
        print(suspects.pop())
    else:
        print(-1)