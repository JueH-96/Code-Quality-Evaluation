import sys

def solve():
    """
    Reads the input, solves the problem, and prints the output.
    """
    # Use fast I/O for performance in competitive programming
    input = sys.stdin.readline

    try:
        # Read the size of the sequence and the sequence itself.
        # The problem statement guarantees valid integer inputs.
        N = int(input())
        A = list(map(int, input().split()))
    except (ValueError, IndexError):
        # Handle cases with empty input, which can occur in some test environments.
        return

    # From the problem constraints, the maximum value of an element is 10^6.
    MAX_VAL = 1_000_000

    # `prefix_sum_le` will be used to store prefix sums.
    # We use a single array for memory efficiency.
    # Step 1: Populate it with sums of elements of a certain value.
    # After this loop, prefix_sum_le[v] will hold the sum of all elements in A equal to v.
    prefix_sum_le = [0] * (MAX_VAL + 1)
    for x in A:
        prefix_sum_le[x] += x

    # Step 2: Convert the array into a cumulative (prefix) sum array.
    # After this loop, prefix_sum_le[v] will hold the sum of all elements in A <= v.
    for v in range(1, MAX_VAL + 1):
        prefix_sum_le[v] += prefix_sum_le[v - 1]

    # The total sum of all elements in A is the sum of all elements up to MAX_VAL.
    total_sum = prefix_sum_le[MAX_VAL]
    
    # Step 3: For each element `a` in the original array A, calculate the result.
    # The sum of elements > a is total_sum - (sum of elements <= a).
    # A list comprehension provides a concise way to generate the results.
    results = [total_sum - prefix_sum_le[a] for a in A]

    # Step 4: Print the final list of results, separated by spaces.
    print(*results)

solve()