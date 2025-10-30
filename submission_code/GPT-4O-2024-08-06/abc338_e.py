# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    chords = []
    
    index = 1
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        if A > B:
            A, B = B, A
        chords.append((A, B))
    
    # Sort chords by their first point
    chords.sort()
    
    # Use a set to keep track of active ends
    active_ends = set()
    
    for A, B in chords:
        # Check if there is any active end that is greater than A and less than B
        # This would mean there is an intersection
        for end in active_ends:
            if A < end < B:
                print("Yes")
                return
        # Add the end of the current chord to active ends
        active_ends.add(B)
    
    print("No")