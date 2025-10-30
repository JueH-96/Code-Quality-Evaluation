import sys

def min_operations(N, P):
    """
    Calculate the minimum number of operations required to satisfy P_i = i for all i.

    Args:
    N (int): The length of the permutation.
    P (list): The permutation.

    Returns:
    int: The minimum number of operations required.
    """
    operations = 0
    for i in range(N - 1, 0, -1):
        if P[i] != i + 1:
            operations += 1
            # Perform the operation with k = i + 1
            # This will move the correct element to the correct position
            # and sort the rest of the list
            P = P[:i] + sorted(P[i:])
            # After the operation, the list is sorted from the (i + 1)th position to the end
            # So we can start checking from the (i - 1)th position in the next iteration
    return operations

T = int(input())
for _ in range(T):
    N = int(input())
    P = list(map(int, input().split()))
    print(min_operations(N, P))