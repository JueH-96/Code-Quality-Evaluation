import sys

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Read sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Read sequence B
    B = list(map(int, sys.stdin.readline().split()))

    # Create a set for quick lookup of elements in A
    # This allows O(1) average time complexity for checking if an element is in A
    A_set = set(A)

    # Combine sequences A and B into a single list C
    C = A + B

    # Sort C in ascending order
    C.sort()

    # Iterate through the sorted list C to find consecutive elements
    # Check if both elements in a consecutive pair are originally from A
    for i in range(len(C) - 1):
        current_element = C[i]
        next_element = C[i+1]

        # If both the current element and the next element are in A_set,
        # then we have found two consecutive elements from A in C.
        if current_element in A_set and next_element in A_set:
            print("Yes")
            return # Exit the function as soon as a pair is found

    # If the loop completes without finding such a pair, print No
    print("No")

# Call the solve function to run the program
solve()