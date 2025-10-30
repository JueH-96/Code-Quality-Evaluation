def solve():
    """
    Reads a list of integers and checks if all of them are equal.
    """
    # Read the number of integers N. While not strictly necessary for this specific
    # solution approach, it's good practice to consume all specified input.
    N = int(input())

    # Read the N space-separated integers into a list of integers.
    A = list(map(int, input().split()))

    # Convert the list to a set to get the unique elements.
    unique_elements = set(A)

    # If the number of unique elements is 1, it means all elements in the
    # original list were the same.
    if len(unique_elements) == 1:
        print("Yes")
    else:
        # If there's more than one unique element, they were not all the same.
        print("No")

# Execute the solution
solve()