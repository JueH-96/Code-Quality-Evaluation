def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    # Convert P values to 0-indexed
    P = [int(x) - 1 for x in data[1:n+1]]
    # Q gives the bib numbers for persons 1 to n (0-indexed here)
    Q = list(map(int, data[n+1:n+1+n]))
    
    # Build a mapping from bib number to the index (0-indexed) of the person wearing that bib.
    bib_to_index = [None] * (n + 1)
    for i, bib in enumerate(Q):
        bib_to_index[bib] = i

    # For each bib number i from 1 to n, determine who the person with that bib is staring at.
    # Let j be the person index such that Q[j] == i.
    # That person j is staring at person P[j] (remember P is 0-indexed).
    # Hence the answer for bib i is Q[P[j]].
    result = [0] * n
    for bib in range(1, n+1):
        j = bib_to_index[bib]
        target = P[j]
        result[bib - 1] = Q[target]
    
    sys.stdout.write(" ".join(map(str, result)))

# Call the main function
main()