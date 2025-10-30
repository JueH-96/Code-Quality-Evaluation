import sys

def solve():
    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().split()
    A_list = [int(x) for x in A_str]

    # Create a list of tuples, where each tuple contains (value, original_1_based_index).
    # This allows us to sort the values while keeping track of their initial positions.
    indexed_A = []
    for i in range(N):
        # A_list[i] is the value, (i + 1) is its 1-based original index.
        indexed_A.append((A_list[i], i + 1))

    # Sort the list of tuples based on the values (the first element of each tuple)
    # in descending order.
    # The 'key=lambda x: x[0]' explicitly tells sort to use the first element of the tuple for comparison.
    # 'reverse=True' ensures descending order.
    indexed_A.sort(key=lambda x: x[0], reverse=True)

    # After sorting, the largest element is at index 0, and the second largest is at index 1.
    # We need the original 1-based index of the second largest element.
    # indexed_A[1] gives us the tuple for the second largest element.
    # The second element of this tuple (indexed_A[1][1]) is its original 1-based index.
    second_largest_info = indexed_A[1]
    original_index_of_second_largest = second_largest_info[1]

    # Print the result.
    print(original_index_of_second_largest)

# Call the solve function to run the program.
solve()