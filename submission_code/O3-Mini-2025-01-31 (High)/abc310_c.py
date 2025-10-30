def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # The first piece of input data is the number of sticks, N
    N = int(input_data[0])
    
    unique_sticks = set()
    # Iterate over each stick's string representation
    for i in range(1, N + 1):
        s = input_data[i]
        # Calculate the reverse of the string
        reversed_s = s[::-1]
        # Canonical representation: the lexicographically smaller string among s and its reverse
        canonical = s if s < reversed_s else reversed_s
        unique_sticks.add(canonical)
    
    # The answer is the number of distinct canonical strings
    sys.stdout.write(str(len(unique_sticks)))
    
# Make sure to call main
if __name__ == '__main__':
    main()