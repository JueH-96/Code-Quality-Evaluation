def main():
    import sys

    # Read the input integer N
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # A helper function to compute the floor of the cube root of n
    def floor_cube_root(n):
        low, high = 1, int(1e6) + 2  # since (10^6)^3 = 10^18 roughly, which covers up to N = 10^18.
        res = 0
        while low <= high:
            mid = (low + high) // 2
            cube = mid * mid * mid
            if cube <= n:
                res = mid
                low = mid + 1
            else:
                high = mid - 1
        return res
    
    # Helper to check if a number is a palindrome
    def is_palindrome(num):
        s = str(num)
        return s == s[::-1]
    
    # Compute the maximum possible x such that x^3 is <= N
    x_max = floor_cube_root(N)
    
    # Iterate backwards from x_max till 1 to find the largest palindromic cube.
    for x in range(x_max, 0, -1):
        cube = x * x * x
        if cube <= N and is_palindrome(cube):
            sys.stdout.write(str(cube))
            return

# Make sure to call main
if __name__ == '__main__':
    main()