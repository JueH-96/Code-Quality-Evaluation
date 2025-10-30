def main():
    import sys
    input = sys.stdin.readline
    # Read input integer N
    N = int(input().strip())
    # Form the string: one L, N os, one n, one g
    dragon_string = "L" + "o" * N + "ng"
    # Output the result
    sys.stdout.write(dragon_string)
    
if __name__ == '__main__':
    main()