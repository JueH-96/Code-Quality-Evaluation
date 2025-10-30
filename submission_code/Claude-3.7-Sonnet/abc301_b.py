def expand_sequence(sequence):
    while True:
        # Check if every pair of adjacent terms has an absolute difference of 1
        all_diff_one = True
        
        for i in range(len(sequence) - 1):
            if abs(sequence[i] - sequence[i+1]) != 1:
                all_diff_one = False
                
                # Determine values to insert
                if sequence[i] < sequence[i+1]:
                    # Insert increasing sequence
                    insert_values = list(range(sequence[i] + 1, sequence[i+1]))
                else:  # sequence[i] > sequence[i+1]
                    # Insert decreasing sequence
                    insert_values = list(range(sequence[i] - 1, sequence[i+1], -1))
                
                # Insert the values into the sequence
                sequence = sequence[:i+1] + insert_values + sequence[i+1:]
                break
        
        if all_diff_one:
            return sequence

def main():
    # Read input
    n = int(input())
    sequence = list(map(int, input().split()))
    
    # Apply the procedure
    final_sequence = expand_sequence(sequence)
    
    # Print the result
    print(' '.join(map(str, final_sequence)))

if __name__ == "__main__":
    main()