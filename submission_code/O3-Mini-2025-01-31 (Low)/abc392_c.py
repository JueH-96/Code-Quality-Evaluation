def main():
    import sys
    input = sys.stdin.readline

    # Reading input values.
    N = int(input().strip())
    P = list(map(int, input().strip().split()))
    Q = list(map(int, input().strip().split()))
    
    # Create an array to map bib number i to the person index having that bib.
    # Since bib numbers are from 1 to N, we can use an array of size N+1.
    person_from_bib = [0] * (N + 1)
    for i in range(N):
        # Person i+1 is wearing bib Q[i]
        bib = Q[i]
        person_from_bib[bib] = i  # store index in 0-based
    
    # For each bib number i (from 1 to N), we find:
    # Let person j be the one wearing bib i. That is, person j index (0-based) is person_from_bib[i].
    # Person j stares at person number P[j] (which is 1-indexed). We need to print the bib number on person P[j].
    result = []
    for bib in range(1, N+1):
        j = person_from_bib[bib]   # j is 0-based index for the person with bib number = bib.
        # The person that person j is staring at:
        target_person_index = P[j] - 1  # convert to 0-based index.
        # The bib number on the target person:
        result.append(str(Q[target_person_index]))
        
    print(" ".join(result))
    
if __name__ == '__main__':
    main()