def main():
    import sys

    # Read input
    N = int(sys.stdin.readline().strip())
    
    # Compute the approximate cube root of N
    # Since floating-point imprecision can occur, we adjust it with a small loop
    x_max = int(N ** (1/3))
    while (x_max + 1) ** 3 <= N:
        x_max += 1
    while x_max ** 3 > N:
        x_max -= 1
    
    # Check descending from x_max down to 1 for palindromic cubes
    for x in range(x_max, 0, -1):
        cube_val = x**3
        s = str(cube_val)
        if s == s[::-1]:
            print(cube_val)
            return

# Do not forget to call main()
if __name__ == "__main__":
    main()