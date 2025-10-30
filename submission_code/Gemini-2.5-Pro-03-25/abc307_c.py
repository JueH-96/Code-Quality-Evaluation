# YOUR CODE HERE
import sys

def solve():
    # Read sheet A dimensions and data
    HA, WA = map(int, sys.stdin.readline().split())
    A = [sys.stdin.readline().strip() for _ in range(HA)]

    # Read sheet B dimensions and data
    HB, WB = map(int, sys.stdin.readline().split())
    B = [sys.stdin.readline().strip() for _ in range(HB)]

    # Read sheet X dimensions and data
    HX, WX = map(int, sys.stdin.readline().split())
    X = [sys.stdin.readline().strip() for _ in range(HX)]

    # Extract relative coordinates of black squares ('#') for sheet A
    # These coordinates are relative to A's top-left corner (0,0)
    PA_rel = set()
    for r in range(HA):
        for c in range(WA):
            if A[r][c] == '#':
                PA_rel.add((r, c))

    # Extract relative coordinates of black squares ('#') for sheet B
    # These coordinates are relative to B's top-left corner (0,0)
    PB_rel = set()
    for r in range(HB):
        for c in range(WB):
            if B[r][c] == '#':
                PB_rel.add((r, c))

    # Extract relative coordinates of black squares ('#') for sheet X
    # These coordinates are relative to X's top-left corner (0,0)
    PX_rel = set()
    for r in range(HX):
        for c in range(WX):
            if X[r][c] == '#':
                PX_rel.add((r, c))

    # Constraints state that A, B, and X each contain at least one black square.
    # Therefore, PX_rel is guaranteed non-empty.
    # Check this condition just in case, although problem says it won't happen.
    if not PX_rel:
        # If X is empty, it could only be matched if A and B are also empty.
        # But constraints say A and B have black squares. So this path leads to "No".
        # However, the core logic handles this correctly anyway if PX_rel is empty 
        # because PC will be non-empty (since PA_rel or PB_rel is non-empty)
        # and len(PC) != len(PX_rel) will be true.
        pass 

    # If PX_rel is non-empty, proceed.
    # Precompute the minimum row and column indices among black squares in X.
    # This helps normalize the shape X for comparison later. Find the 'anchor' point.
    # Since PX_rel is guaranteed non-empty, min() will work.
    min_rX_rel = min(r for r, c in PX_rel)
    min_cX_rel = min(c for r, c in PX_rel)
    
    # Determine the search range for the relative displacement (drB, dcB) of sheet B with respect to sheet A.
    # Sheet A is assumed to be placed with its top-left at (0,0). Sheet B's top-left is at (drB, dcB).
    # Analysis based on the problem constraints (dimensions <= 10) shows that the required displacement 
    # is bounded. A range of [-20, 20] is conservative and safe.
    RANGE = 20  # Defines the search window from -20 to +20 for both row and column shifts.

    # Iterate through all possible relative placements (drB, dcB) for sheet B.
    for drB in range(-RANGE, RANGE + 1):
        for dcB in range(-RANGE, RANGE + 1):
            
            # Compute the set of absolute coordinates PC for the combined black squares from A and B
            # on the infinite sheet C.
            # Start with coordinates from A (its relative coords are absolute coords since A is placed at origin)
            PC = set(PA_rel) 

            # Calculate absolute coordinates for B's black squares based on the current shift (drB, dcB)
            PB_abs = set()
            for r_rel, c_rel in PB_rel:
                 PB_abs.add((r_rel + drB, c_rel + dcB))
            
            # Update PC with the points from B. The set union automatically handles any overlaps.
            PC.update(PB_abs)

            # If the combined shape PC is empty, it cannot match non-empty X. Skip.
            # This check is mostly for conceptual clarity as constraints guarantee non-empty A, B, X.
            if not PC:
                 continue

            # The number of black squares in the combined shape PC must exactly match the number in X.
            # If the counts differ, the shapes cannot be identical.
            if len(PC) != len(PX_rel):
                continue

            # Find the minimum row and column indices among black squares in the combined shape PC.
            # This identifies the top-leftmost extent of the combined shape.
            min_rC = min(r for r, c in PC)
            min_cC = min(c for r, c in PC)

            # Calculate the required translation (drX_shift, dcX_shift) that would align
            # the shape defined by PX_rel (X's pattern) with the combined shape PC, assuming they could match.
            # This shift aligns the precomputed minimum coordinate point of X's pattern with PC's minimum point.
            drX_shift = min_rC - min_rX_rel
            dcX_shift = min_cC - min_cX_rel
            
            # Construct the set of absolute coordinates PX_translated by applying the calculated shift
            # to the relative coordinates of X's black squares. This represents what shape X would look like
            # if placed on sheet C such that its pattern aligns with PC's top-leftmost extent.
            PX_translated = set()
            for r_rel, c_rel in PX_rel:
                PX_translated.add((r_rel + drX_shift, c_rel + dcX_shift))

            # Compare the set of coordinates of the combined shape PC with the translated shape of X (PX_translated).
            # If these two sets are identical, it means we found a relative placement (drB, dcB) for B
            # such that the combined shape formed by A and B exactly matches the shape of X, potentially translated.
            # This implies a valid placement and cut-out exist.
            if PC == PX_translated:
                 # Found a valid configuration. Print "Yes" and terminate the program.
                 print("Yes")
                 return

    # If the loops complete without finding any valid configuration after checking all possibilities
    # within the defined range, it means it's impossible to achieve the goal.
    print("No")

# Execute the solve function to run the program
solve()