def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N, M, P = map(int, input_data)
    
    # If the first full moon day M is after day N, then no full moons are seen.
    if M > N:
        print(0)
    else:
        # Count how many full moon days from day M to day N (inclusive).
        # The sequence is: M, M + P, M + 2P, ... <= N.
        # The number of terms is ((N - M) // P) + 1.
        count = ((N - M) // P) + 1
        print(count)

if __name__ == '__main__':
    main()