def main():
    import sys
    
    # Read input
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    
    # Filter multiples of K, divide them by K
    result = [str(a // K) for a in A if a % K == 0]
    
    # Print the quotients in ascending order
    print(" ".join(result))


if __name__ == "__main__":
    main()