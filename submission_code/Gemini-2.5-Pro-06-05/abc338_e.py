import sys

def solve():
    """
    Solves the chord intersection problem by modeling it as a parenthesis matching problem.
    """
    try:
        # For fast I/O in Python
        input = sys.stdin.readline
        
        # Read the number of chords
        N = int(input())
        
        # partner[p] will store the point connected to p.
        # Array is of size 2*N+1 for 1-based indexing.
        partner = [0] * (2 * N + 1)
        for _ in range(N):
            a, b = map(int, input().split())
            partner[a] = b
            partner[b] = a

        # Stack to keep track of the "opening" points of chords.
        stack = []
        
        # Iterate through all points in clockwise order.
        for p in range(1, 2 * N + 1):
            other_end = partner[p]
            
            # If p is the smaller endpoint of its chord, it's an "opening" event.
            if p < other_end:
                stack.append(p)
            # If p is the larger endpoint, it's a "closing" event.
            else:
                # A closing event must match the last opening event for chords to not intersect.
                # If the stack is empty or the top doesn't match, we have an intersection.
                # Note: For valid inputs, the stack won't be empty here.
                last_opened = stack.pop()
                if last_opened != other_end:
                    print("Yes")
                    return
        
        # If the loop completes, all chords are properly nested without intersection.
        print("No")
        
    except (IOError, ValueError):
        # Handle potential errors with input reading, though not expected on judging platforms.
        # This could happen with empty input locally.
        pass

solve()