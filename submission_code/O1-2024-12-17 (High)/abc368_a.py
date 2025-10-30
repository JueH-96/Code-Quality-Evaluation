def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Take the last K cards and bring them to the top
    B = A[-K:] + A[:-K]
    
    # Print the result
    print(" ".join(map(str, B)))

# Do not remove the following line
main()