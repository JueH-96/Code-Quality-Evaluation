def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))

    # We will store the operations in a list of tuples
    operations = []

    # We will use the fact that we can swap elements to move each element to its correct position
    for i in range(N):
        # If the element at the current position is not i+1, we need to swap it
        if A[i] != i + 1:
            # Find the index of the element that should be at the current position
            correct_index = A.index(i + 1)
            # Swap the elements
            A[i], A[correct_index] = A[correct_index], A[i]
            # Record the operation
            operations.append((i + 1, correct_index + 1))

    # Output the number of operations
    print(len(operations))
    # Output each operation
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()