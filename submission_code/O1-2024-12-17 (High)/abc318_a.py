def main():
    import sys

    N, M, P = map(int, sys.stdin.readline().split())
    
    if M > N:
        print(0)
        return
    
    # The full moon occurs on days M, M+P, M+2P, ... up to <= N.
    # Calculate how many such days are within [1..N].
    count = 1 + (N - M) // P
    print(count)

# Don't forget to call the main function
if __name__ == "__main__":
    main()