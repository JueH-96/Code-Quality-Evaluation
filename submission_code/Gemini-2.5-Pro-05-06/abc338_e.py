import sys

def main():
    N = int(sys.stdin.readline())
    
    # partner[i] stores the point connected to point i by a chord
    # Points are 1-indexed, so use an array of size 2N+1
    partner = [0] * (2 * N + 1)
    
    for _ in range(N):
        u, v = map(int, sys.stdin.readline().split())
        partner[u] = v
        partner[v] = u

    # Stack to keep track of the closing points of opened chords.
    # When a chord (s, e) with s < e is opened at point s, we push e onto stack.
    stack = []
    
    # Iterate through points on the circle in clockwise order (1 to 2N)
    for p in range(1, 2 * N + 1):
        # partner_of_p is the other endpoint of the chord involving p
        partner_of_p = partner[p]
        
        # Case 1: p is the "left" endpoint of its chord (i.e., p < partner_of_p)
        if p < partner_of_p:
            # This is an opening point for chord (p, partner_of_p).
            # Push its corresponding closing point (which is partner_of_p) onto the stack.
            stack.append(partner_of_p)
        # Case 2: p is the "right" endpoint of its chord (i.e., p > partner_of_p)
        else:
            # This is a closing point for chord (partner_of_p, p).
            # When partner_of_p was processed (since partner_of_p < p), 
            # p itself should have been pushed onto the stack.
            # So, p should be at the top of the stack if no intersections have "crossed" this chord.
            
            # The stack should not be empty here if input is valid (N chords, 2N distinct points).
            # If partner_of_p (< p) was processed, it must have pushed p.
            # An empty stack at this point would indicate a malformed structure or error.
            # However, standard problem constraints usually ensure this won't happen.
            # If it were possible, `stack.pop()` would raise an IndexError.
            # For robustness, one might check `if not stack: ...`, but it's not strictly needed here.

            # Pop the top element from the stack. This element is the closing point
            # of the chord that was most recently opened and is still active (i.e., "innermost").
            top_val_from_stack = stack.pop()
            
            # We are currently at point p. This point p is closing the chord (partner_of_p, p).
            # Therefore, p itself is the value that should have been on top of the stack
            # if this chord (partner_of_p, p) was indeed the innermost one to be closed.
            if top_val_from_stack != p:
                # An intersection is found.
                # This means that the chord whose closing point was top_val_from_stack (innermost on stack)
                # is different from the chord (partner_of_p, p) that current point p is closing.
                # This configuration implies an intersection (e.g., s1 < s2 < e1 < e2).
                print("Yes")
                return
                
    # If the loop completes, all points have been processed.
    # If no intersections were found, print "No".
    # The stack should be empty at this point if all chords closed correctly.
    print("No")

if __name__ == '__main__':
    main()