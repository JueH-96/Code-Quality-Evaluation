import sys
import threading

def main():
    import sys
    
    input = sys.stdin.readline
    N = int(input())
    
    # We will map each point on the circle (1..2N) to a chord index.
    # Also keep track whether that occurrence is the first time we see this chord (start)
    # or the second time (end).
    # Then we scan points 1..2N in order, and use a stack to check for crossings.
    
    # pair_pos will store for each chord i the two endpoints (a, b) sorted so a < b
    pair_pos = [None] * (N + 1)
    # endpoint_to_chord[p] = (chord_index, is_start)
    # We'll fill chord_index for each position, and determine start/end on the fly.
    endpoint_to_chord = [0] * (2 * N + 1)
    
    for i in range(1, N + 1):
        a, b = map(int, input().split())
        if a < b:
            pair_pos[i] = (a, b)
        else:
            pair_pos[i] = (b, a)
        # Mark both positions to this chord
        endpoint_to_chord[pair_pos[i][0]] = i
        endpoint_to_chord[pair_pos[i][1]] = i
    
    # seen[i] will tell us if we've already encountered chord i once (the start).
    seen = [False] * (N + 1)
    stack = []
    
    # Iterate around the circle in clockwise order from point 1 to point 2N
    for pos in range(1, 2 * N + 1):
        cid = endpoint_to_chord[pos]
        if not seen[cid]:
            # first time we see chord cid => it's the "start"
            seen[cid] = True
            stack.append(cid)
        else:
            # second time => it's the "end"
            # If the chord ending now is not on top of the stack, we have a crossing
            if not stack or stack[-1] != cid:
                print("Yes")
                return
            # otherwise it's a proper nested match
            stack.pop()
    
    # If we never found a mismatch, there is no crossing
    print("No")

if __name__ == "__main__":
    main()