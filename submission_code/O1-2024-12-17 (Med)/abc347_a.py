def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Filter elements in A that are multiples of K
    multiples = [x for x in A if x % K == 0]

    # Compute the quotients
    quotients = [x // K for x in multiples]

    # Since A is strictly increasing, quotients will already be in ascending order
    # Print the quotients with spaces in between
    print(" ".join(map(str, quotients)))

# Do not forget to call main()
main()