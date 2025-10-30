def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    N = int(input_data[0])
    T = int(input_data[1])
    P = int(input_data[2])
    L = list(map(int, input_data[3:3+N]))

    # For each person, calculate the days required to reach at least T.
    required_days = []
    for hair in L:
        needed = T - hair
        # If hair is already at least T, 0 days are needed.
        if needed < 0:
            needed = 0
        required_days.append(needed)
    
    # Sort the required days.
    required_days.sort()

    # The answer is the P-th smallest required days.
    # This ensures that at least P people's hair reach T after this many days.
    print(required_days[P-1])

if __name__ == '__main__':
    main()