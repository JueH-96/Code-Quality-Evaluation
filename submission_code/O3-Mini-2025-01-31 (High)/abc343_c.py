def is_palindrome(num):
    s = str(num)
    return s == s[::-1]

def integer_cube_root(n):
    # Binary search for the largest integer x such that x^3 <= n.
    lo, hi = 1, int(1e6)  # Because (10^6)^3 = 10^18, which is the maximum possible cube root.
    ans = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        cube = mid * mid * mid
        if cube <= n:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans

def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    # Compute the maximum possible x such that x^3 <= n.
    x_max = integer_cube_root(n)
    
    # Iterate from x_max down to 1, checking if x^3 is palindromic.
    # The first encountered (largest) palindromic cube will be our answer.
    ans = 0
    for x in range(x_max, 0, -1):
        cube = x * x * x
        if cube <= n and is_palindrome(cube):
            ans = cube
            break

    print(ans)

if __name__ == '__main__':
    main()