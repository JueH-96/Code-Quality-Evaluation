def intersect(a, b, c, d):
    """ Helper function to check if two chords (a, b) and (c, d) intersect """
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    return (a < c < b < d) or (c < a < d < b)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    chords = []
    
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        chords.append((A, B))
        index += 2
    
    # Sort chords by their starting point
    chords.sort(key=lambda x: min(x))
    
    # Use a stack to keep track of active chords (using their endpoints)
    stack = []
    
    for a, b in chords:
        if a > b:
            a, b = b, a
        
        # Remove chords from the stack that are completely before the current chord's start
        while stack and stack[-1] < a:
            stack.pop()
        
        # Check if the current chord intersects with any chord in the stack
        for endpoint in stack:
            if endpoint > b:
                print("Yes")
                return
        
        # Add the endpoint of the current chord to the stack
        stack.append(b)
        stack.sort(reverse=True)  # Keep the stack sorted by endpoints in descending order
    
    print("No")

main()