# YOUR CODE HERE
def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Calculate the Least Common Multiple of a and b."""
    return a * b // gcd(a, b)

def count_exact_one(n, m, x):
    """
    Calculate the number of integers from 1 to x that are divisible by exactly one of n and m.
    """
    n_div = x // n
    m_div = x // m
    nm_div = x // lcm(n, m)
    return n_div + m_div - 2 * nm_div

def find_kth_number(n, m, k):
    """
    Find the k-th smallest positive integer that is divisible by exactly one of n and m.
    """
    # Binary search to find the smallest x such that count(x) >= K
    low, high = 1, 10**18  # Setting a high upper bound
    while low < high:
        mid = (low + high) // 2
        count = count_exact_one(n, m, mid)
        if count < k:
            low = mid + 1
        else:
            high = mid
    
    return low

# Read input
n, m, k = map(int, input().split())

# Find the k-th number
print(find_kth_number(n, m, k))