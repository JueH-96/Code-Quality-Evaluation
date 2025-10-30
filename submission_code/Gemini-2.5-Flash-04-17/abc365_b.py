import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Create a sorted copy to find the second largest value
    sorted_A = sorted(A)

    # The second largest value is the second to last element in the sorted list
    second_largest_value = sorted_A[-2]

    # Find the 0-based index of the second largest value in the original list A
    # Since elements are distinct, A.index() will return the correct unique index
    original_index_0based = A.index(second_largest_value)

    # The problem asks for the 1-based index (X-th element)
    result_index_1based = original_index_0based + 1

    # Print the result
    print(result_index_1based)

solve()