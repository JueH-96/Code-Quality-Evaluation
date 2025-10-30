import sys

def main():
    """
    This function implements the solution logic.
    """
    # Fast I/O
    readline = sys.stdin.readline
    
    # Read N and X
    try:
        line = readline()
        if not line: return
        N, X = map(int, line.split())
        
        # Read the sequence A
        A = list(map(int, readline().split()))
    except (IOError, ValueError):
        # In case of malformed input, though not expected by problem spec
        print("-1")
        return

    # `pos[v]` will store a list of 1-based indices for value `v`.
    # We only need to store up to 3 indices for any value because we are looking
    # for a triple, and the worst case is needing three of the same value.
    pos = [[] for _ in range(X + 1)]
    for i, v in enumerate(A):
        if v <= X:
            if len(pos[v]) < 3:
                pos[v].append(i + 1)
    
    # Get a sorted list of unique values present in A.
    unique_vals = [v for v in range(1, X + 1) if pos[v]]

    # Iterate through all pairs of unique values (v1, v2) such that v1 <= v2.
    for i in range(len(unique_vals)):
        v1 = unique_vals[i]
        for j in range(i, len(unique_vals)):
            v2 = unique_vals[j]
            
            # Calculate the third value required.
            v3 = X - v1 - v2

            # To avoid permutations and redundant checks, we enforce v1 <= v2 <= v3.
            # Since the outer loops ensure v1 <= v2, we only need to check v3 >= v2.
            if v3 < v2:
                continue
            
            # Check if v3 is a valid value and exists in the array A.
            if v3 > X or not pos[v3]:
                continue

            # At this point, we have found a valid triple of values (v1, v2, v3).
            # Now, we need to find distinct indices for them.
            
            indices = []
            
            if v1 == v2 and v2 == v3: # Case: v1 = v2 = v3
                # We need three occurrences of the same value.
                if len(pos[v1]) >= 3:
                    indices.extend(pos[v1][:3])
                else:
                    continue # Not enough occurrences.
            elif v1 == v2: # Case: v1 = v2 < v3
                # We need two of v1 and one of v3.
                if len(pos[v1]) >= 2:
                    indices.extend(pos[v1][:2])
                    indices.append(pos[v3][0])
                else:
                    continue
            elif v2 == v3: # Case: v1 < v2 = v3
                # We need one of v1 and two of v2.
                if len(pos[v2]) >= 2:
                    indices.append(pos[v1][0])
                    indices.extend(pos[v2][:2])
                else:
                    continue
            else: # Case: v1 < v2 < v3 (all distinct)
                # We need one of each.
                indices.append(pos[v1][0])
                indices.append(pos[v2][0])
                indices.append(pos[v3][0])

            # If we found a valid set of indices, print them sorted and exit.
            print(*sorted(indices))
            return
    
    # If the loops complete without finding a solution, print -1.
    print(-1)

if __name__ == "__main__":
    main()