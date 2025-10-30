def main():
    import sys
    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # We're looking for the maximum palindromic cube <= N.
    # Since N <= 10^18, the largest x such that x^3 <= 10^18 is x = 10^6.
    
    max_palindromic_cube = 0
    
    # We'll check each x up to 1,000,000.
    for x in range(1, 1_000_001):
        cube_val = x**3
        if cube_val > N:
            break
        # Check if cube_val is a palindrome by string check:
        s = str(cube_val)
        if s == s[::-1]:
            max_palindromic_cube = cube_val
    
    print(max_palindromic_cube)


if __name__ == "__main__":
    main()