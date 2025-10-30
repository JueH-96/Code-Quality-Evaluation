def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read N and Q
    N = int(data[0])
    Q = int(data[1])
    
    # Initialize an array with True (tooth present) for each of the N holes.
    teeth = [True] * N
    
    # Process each treatment
    for i in range(Q):
        # Each treatment is applied to hole T_i (1-indexed).
        t = int(data[2 + i])
        index = t - 1  # Convert to 0-indexed
        # Toggle the state of the tooth: if True, remove it (set to False); if False, grow a tooth (set to True).
        teeth[index] = not teeth[index]
    
    # Count the number of teeth after all treatments.
    result = sum(teeth)
    
    # Print the result
    print(result)

# Call the main function
if __name__ == '__main__':
    main()