def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Find the largest value in A.
    # Since 1 <= A_i <= 100 and N >= 2, A is non-empty and max(A) is well-defined.
    max_val = max(A)

    # Create a list of elements from A that are not equal to max_val.
    # These are the "integers that are not the largest".
    not_largest_elements = [x for x in A if x != max_val]

    # The problem constraint "It is not the case that all A_1, A_2, ..., A_N are equal"
    # guarantees that not_largest_elements will not be empty.
    # For example, if A = [3,3,3], then max_val = 3, and not_largest_elements would be [].
    # Then max([]) would raise an error. However, this case [3,3,3] is disallowed by constraints.
    # If A = [3,3,2], then max_val = 3, not_largest_elements = [2]. max([2]) = 2.

    # Find the largest among these 'not_largest_elements'.
    result = max(not_largest_elements)

    # Print the result.
    print(result)

if __name__ == '__main__':
    solve()