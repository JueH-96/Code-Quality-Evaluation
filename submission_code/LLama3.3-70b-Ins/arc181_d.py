import sys

def calculate_inversion_number(P):
    """
    Calculate the inversion number of a sequence.
    
    Args:
    P (list): A list of integers representing the sequence.
    
    Returns:
    int: The inversion number of the sequence.
    """
    inversion_number = 0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if P[i] > P[j]:
                inversion_number += 1
    return inversion_number

def perform_operation(P, k):
    """
    Perform operation k on the permutation P.
    
    Args:
    P (list): A list of integers representing the permutation.
    k (int): The operation number.
    
    Returns:
    list: The modified permutation after performing the operation.
    """
    for i in range(k - 1):
        if P[i] > P[i + 1]:
            P[i], P[i + 1] = P[i + 1], P[i]
    return P

def main():
    # Read input from stdin
    N = int(input())
    P = list(map(int, input().split()))
    M = int(input())
    A = list(map(int, input().split()))

    # Perform operations and calculate inversion numbers
    for i in range(M):
        for j in range(i + 1):
            P = perform_operation(P, A[j])
        inversion_number = calculate_inversion_number(P)
        print(inversion_number)

if __name__ == "__main__":
    main()