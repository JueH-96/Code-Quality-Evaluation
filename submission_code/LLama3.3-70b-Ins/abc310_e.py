def calculate_sum(n, s):
    """
    Calculate the sum of f(i, j) for all pairs (i, j) where 1 <= i <= j <= n.

    Args:
    n (int): The length of the string s.
    s (str): A string of length n consisting of 0 and 1.

    Returns:
    int: The sum of f(i, j) for all pairs (i, j) where 1 <= i <= j <= n.
    """
    total_sum = 0
    for i in range(n):
        for j in range(i, n):
            # Initialize f(i, j) with A_i
            f_ij = int(s[i])
            # Calculate f(i, j) using the recursive formula
            for k in range(i + 1, j + 1):
                # Calculate f(i, k) using the NAND operator
                f_ij = int(not (f_ij and int(s[k])))
            # Add f(i, j) to the total sum
            total_sum += f_ij
    return total_sum

# Read the input from stdin
n = int(input())
s = input()

# Calculate and print the sum
print(calculate_sum(n, s))