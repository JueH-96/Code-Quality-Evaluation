def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    Q = int(input_data[1])
    # Initialize teeth presence: True means tooth present.
    # We use 0-indexed list, while holes are 1-indexed.
    teeth = [True] * N
    
    # Process each treatment
    for i in range(Q):
        t = int(input_data[2 + i]) - 1  # convert to 0-index
        # Toggle the state of tooth in hole t
        teeth[t] = not teeth[t]
    
    # Count how many teeth remain
    result = sum(teeth)
    print(result)

if __name__ == '__main__':
    main()