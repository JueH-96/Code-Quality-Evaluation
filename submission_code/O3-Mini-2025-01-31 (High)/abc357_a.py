def main():
    import sys
    input_data = sys.stdin.read().split()
    # Parse first two numbers: N aliens and M capacity (number of hand disinfectants)
    N = int(input_data[0])
    M = int(input_data[1])
    
    # The next N numbers represent the number of hands for each alien.
    hands = list(map(int, input_data[2:2+N]))
    
    count = 0  # Counter for aliens who disinfect all their hands
    remaining = M  # Remaining disinfectant capacity
    
    for h in hands:
        # If there is enough disinfectant to cover all h hands
        if remaining >= h:
            remaining -= h
            count += 1
        else:
            # If not enough disinfectant for this alien, they use what remains and stop
            break

    sys.stdout.write(str(count))

# Ensure main is called when the program runs
if __name__ == '__main__':
    main()