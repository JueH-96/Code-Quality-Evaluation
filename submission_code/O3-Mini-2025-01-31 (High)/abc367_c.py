def main():
    import sys

    # Read input data from standard input
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:2+N]))

    # We'll collect output lines to print them at the end.
    output_lines = []
    
    # This variable will hold the current sequence we are generating.
    sequence = []
    
    # Backtracking function to generate sequences in lexicographical order.
    def backtrack(index, current_sum):
        # When we have chosen N elements, check the sum condition.
        if index == N:
            if current_sum % K == 0:
                output_lines.append(" ".join(map(str, sequence)))
            return
        
        # For the index-th position choose every number from 1 up to R[index] (inclusive)
        for num in range(1, R[index] + 1):
            sequence.append(num)
            backtrack(index + 1, current_sum + num)
            sequence.pop()
    
    # Kick off backtracking from position 0 with initial sum 0.
    backtrack(0, 0)
    
    # Print the generated sequences.
    sys.stdout.write("
".join(output_lines))
    
# Call the main function to execute the program.
if __name__ == '__main__':
    main()