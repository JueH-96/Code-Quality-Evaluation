def solve():
    """
    Reads a single test case, finds the optimal solution, and prints it.
    """
    # Read the size of the array (n) and the array of digits (a).
    # n is not strictly necessary as len(a) gives the size, but we read it
    # to follow the input format.
    n = int(input())
    a = list(map(int, input().split()))

    # The optimal strategy is to add 1 to the smallest digit.
    # We find the index of the minimum element. If there are duplicates of the
    # minimum value, list.index() will return the index of the first one,
    # which is sufficient.
    min_idx = 0
    for i in range(1, n):
        if a[i] < a[min_idx]:
            min_idx = i

    # Increment the smallest digit by 1 in-place.
    a[min_idx] += 1

    # Calculate the product of all elements in the modified array.
    product = 1
    for x in a:
        product *= x
    
    # Print the result.
    print(product)

def main():
    """
    Main function to handle the test case loop.
    """
    # Read the number of test cases.
    try:
        t = int(input())
    except (ValueError, EOFError):
        t = 0
    
    # Process each test case.
    for _ in range(t):
        solve()

if __name__ == "__main__":
    main()