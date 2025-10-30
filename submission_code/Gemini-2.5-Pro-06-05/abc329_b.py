# YOUR CODE HERE
def solve():
    """
    Reads a list of integers and finds the largest among those that are not the largest.
    """
    # Read N, though it's not strictly necessary for this specific solution,
    # it's part of the problem's input format.
    try:
        n_str = input()
        if not n_str: return
        n = int(n_str)
        
        # Read the list of N integers from a single line of input.
        a = list(map(int, input().split()))

        # 1. Get unique elements by converting the list to a set.
        # 2. Convert the set back to a list.
        # 3. Sort the list of unique elements.
        unique_sorted = sorted(list(set(a)))
        
        # The problem asks for the largest number that is not THE largest.
        # This is the same as the second-largest unique number. In a sorted
        # list of unique numbers, this is the second-to-last element (index -2).
        # The problem constraints guarantee that there are at least two unique numbers.
        print(unique_sorted[-2])

    except (IOError, ValueError):
        # Handle potential empty input or invalid format, though not expected by constraints.
        pass

solve()