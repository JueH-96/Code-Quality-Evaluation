def main():
    import sys
    input_line = sys.stdin.read().strip()
    if input_line:
        tokens = input_line.split()
        A = int(tokens[0])
        B = int(tokens[1])
        D = int(tokens[2])
        
        # Create the arithmetic sequence using a range (inclusive of B)
        sequence = list(range(A, B + 1, D))
        
        # Print the sequence as required (space separated)
        print(" ".join(map(str, sequence)))

if __name__ == '__main__':
    main()