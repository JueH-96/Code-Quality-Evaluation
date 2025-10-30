def cube_root(N):
    """Finds the largest integer x such that x^3 <= N using binary search."""
    low = 1
    high = N
    while low <= high:
        mid = (low + high) // 2
        mid_cubed = mid * mid * mid
        if mid_cubed == N:
            return mid
        elif mid_cubed < N:
            low = mid + 1
        else:
            high = mid - 1
    return high

def is_palindrome(K):
    """Checks if the integer K is a palindrome."""
    s = str(K)
    return s == s[::-1]

def main():
    import sys
    N_str = sys.stdin.read().strip()
    if not N_str:
        print(1)
        return
    N = int(N_str)
    x_max = cube_root(N)
    for x in range(x_max, 0, -1):
        K = x * x * x
        if is_palindrome(K):
            print(K)
            return

if __name__ == "__main__":
    main()