import sys

def solve():
    # Read N and K from the first line
    n, k = map(int, sys.stdin.readline().split())

    # Read the sequence A from the second line
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize an empty list to store the quotients
    quotients = []

    # Iterate through each element in A
    for x in a:
        # Check if the element is a multiple of K
        if x % k == 0:
            # If it is, calculate the quotient and add it to the list
            quotients.append(x // k)

    # The problem statement guarantees A is sorted (A_1 < A_2 < ... < A_N).
    # Since K is positive, if A_i < A_j and both are multiples of K,
    # then A_i // K < A_j // K.
    # Therefore, the 'quotients' list will naturally be in ascending order
    # without needing an explicit sort.

    # Print the quotients, separated by spaces
    # Convert each integer quotient back to a string for joining
    print(*(str(q) for q in quotients))

if __name__ == '__main__':
    solve()