def main():
    S = input().strip()
    T = input().strip()
    
    # If S and T are already equal, no operations needed
    if S == T:
        print(0)
        return
    
    operations = []
    current_string = S
    
    # Continue until S becomes T
    while current_string != T:
        # Find all positions where characters differ
        diff_positions = [i for i in range(len(current_string)) if current_string[i] != T[i]]
        
        # Try changing each position and find the lexicographically smallest result
        min_string = None
        for pos in diff_positions:
            # Create new string by changing the character at position pos
            new_string = current_string[:pos] + T[pos] + current_string[pos+1:]
            
            # Update if this is lexicographically smaller than our current best
            if min_string is None or new_string < min_string:
                min_string = new_string
        
        # Add the best transformation to our operations list
        operations.append(min_string)
        current_string = min_string
    
    # Output the results
    print(len(operations))
    for op in operations:
        print(op)

if __name__ == "__main__":
    main()