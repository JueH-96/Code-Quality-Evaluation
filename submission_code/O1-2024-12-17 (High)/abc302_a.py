def main():
    import sys
    A, B = map(int, sys.stdin.read().strip().split())
    # The required number of attacks is ceil(A / B), which can be computed as (A + B - 1) // B
    print((A + B - 1) // B)

# Do not forget to call main() at the end of the program
main()