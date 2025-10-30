def calculate_remainder(N):
    """
    Calculate the remainder when V_N is divided by 998244353.

    Args:
    N (int): A positive integer.

    Returns:
    int: The remainder when V_N is divided by 998244353.
    """
    MOD = 998244353
    # Convert N to a string to calculate its length
    str_N = str(N)
    # Calculate the length of N
    len_N = len(str_N)
    # Calculate the value of V_N
    V_N = int(str_N * N)
    # Calculate the remainder when V_N is divided by MOD
    remainder = V_N % MOD
    return remainder

# Read the input from stdin
N = int(input())

# Calculate and print the remainder
remainder = calculate_remainder(N)
print(remainder)