def main():
    import sys
    # Read the entire input and split it into tokens.
    data = sys.stdin.read().split()
    
    # The first token is N.
    n = int(data[0])
    
    # The next N tokens are the persons each person is staring at, P.
    # They are given 1-indexed so we'll maintain that convention when needed.
    P = list(map(int, data[1:n+1]))
    
    # The following N tokens are bib numbers Q.
    Q = list(map(int, data[n+1:2*n+1]))
    
    # Build a mapping from bib number to the person index (0-indexed) who wears that bib.
    # Since Q is a permutation of {1, 2, ..., N}, we can simply store the index for each bib.
    bib_to_index = [0] * (n + 1)
    for i in range(n):
        bib_to_index[Q[i]] = i  # storing the person index (0-indexed) for bib Q[i]
    
    # For each bib number i from 1 to N, determine the bib number of the person being stared at.
    # Person with bib i is found at index bib_to_index[i]
    # That person is staring at person P[j] (which is 1-indexed), so we get Q[P[j]-1].
    result = []
    for bib in range(1, n + 1):
        person_index = bib_to_index[bib]     # this is j where Q[j] == bib
        target_person = P[person_index]        # person j is staring at target_person (1-indexed)
        target_bib = Q[target_person - 1]        # get the bib of the target person
        result.append(str(target_bib))
    
    # Output the result as space-separated values.
    sys.stdout.write(" ".join(result))

# Call the main function to execute the solution.
if __name__ == "__main__":
    main()