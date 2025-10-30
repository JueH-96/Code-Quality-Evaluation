import sys

def min_operations(A):
    """
    Calculate the minimum number of operations required to make the difference between the minimum and maximum values of A at most one.

    Args:
    A (list): A list of integers.

    Returns:
    int: The minimum number of operations required.
    """
    A.sort()  # Sort the list in ascending order
    operations = 0

    # Calculate the target value
    target = sum(A) // len(A)

    # Calculate the number of operations required
    for num in A:
        if num > target + 1:
            operations += num - (target + 1)
        elif num < target:
            operations += target - num

    return operations

def main():
    # Read input from stdin
    N = int(input())
    A = list(map(int, input().split()))

    # Calculate and print the minimum number of operations
    print(min_operations(A))

if __name__ == "__main__":
    main()