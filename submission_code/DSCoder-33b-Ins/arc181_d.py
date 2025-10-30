# YOUR CODE HERE

def solve():
    N = int(input())
    P = list(map(int, input().split()))
    M = int(input())
    A = list(map(int, input().split()))

    # Create a list to store the inversion number of P after each operation
    inversion_numbers = []

    # Create a list to store the permutation after each operation
    permutations = [P]

    # Perform operations
    for a in A:
        # Get the last permutation
        P = permutations[-1]

        # Perform operation a
        for i in range(a-1):
            if P[i] > P[i+1]:
                P[i], P[i+1] = P[i+1], P[i]

        # Store the permutation after operation a
        permutations.append(P)

    # Calculate the inversion number of each permutation
    for P in permutations:
        inversion_number = 0
        for i in range(N):
            for j in range(i+1, N):
                if P[i] > P[j]:
                    inversion_number += 1
        inversion_numbers.append(inversion_number)

    # Print the inversion numbers
    for inversion_number in inversion_numbers:
        print(inversion_number)

solve()