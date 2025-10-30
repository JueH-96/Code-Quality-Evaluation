# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Create a dictionary to map person index to their bib number
    bib_number_by_person = {i+1: Q[i] for i in range(N)}
    
    # Create a result list to store the bib numbers of the person each person is staring at
    result = []
    
    for i in range(1, N+1):
        # Find the person i is staring at
        staring_at_person = P[i-1]
        # Find the bib number of the person i is staring at
        bib_number = bib_number_by_person[staring_at_person]
        result.append(bib_number)
    
    # Print the result as a space-separated string
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()