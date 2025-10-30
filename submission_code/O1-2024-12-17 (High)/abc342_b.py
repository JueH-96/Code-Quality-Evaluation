# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = [int(x) for x in data[1:1+N]]
    Q = int(data[1+N])
    
    # For quick lookups, create a dictionary to get the index of each person.
    # person_position[x] will give the index of person x in the list P.
    person_position = {person: idx for idx, person in enumerate(P)}
    
    idx = 0
    for _ in range(Q):
        A, B = int(data[2+N+2*idx]), int(data[2+N+2*idx + 1])
        idx += 1
        # Compare the positions of A and B in P
        if person_position[A] < person_position[B]:
            print(A)
        else:
            print(B)

# Do not forget to call main()
main()