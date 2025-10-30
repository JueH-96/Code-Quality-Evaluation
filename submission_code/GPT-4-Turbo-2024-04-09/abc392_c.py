import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Create a mapping from person index to bib number
    bib_number = {}
    for index in range(N):
        bib_number[index + 1] = Q[index]
    
    # Create the result list
    result = [0] * N
    
    # For each person i, find the bib number of the person they are staring at
    for i in range(1, N + 1):
        person_stared_at = P[i - 1]
        bib_of_person_stared_at = bib_number[person_stared_at]
        result[i - 1] = bib_of_person_stared_at
    
    # Print the result
    print(" ".join(map(str, result)))