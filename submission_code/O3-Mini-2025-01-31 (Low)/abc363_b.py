def main():
    import sys
    input_line = sys.stdin.readline().strip()
    if not input_line:
        return
    N, T, P = map(int, input_line.split())
    hair_lengths = list(map(int, sys.stdin.readline().split()))

    # Calculate days needed for each person to reach at least T.
    days_needed = [max(0, T - L) for L in hair_lengths]
    
    # Sort to find the P-th smallest duration.
    days_needed.sort()
    # The P-th person in sorted order determines when we have P people with hair length >= T.
    print(days_needed[P-1])

if __name__ == "__main__":
    main()