def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    N = int(data[1])
    
    # Calculate the number of possible combinations
    num_questions = S.count('?')
    max_value = -1
    
    # Iterate over all possible replacements of '?' with '0' or '1'
    for i in range(2 ** num_questions):
        # Generate a binary string of length num_questions
        bin_str = bin(i)[2:].zfill(num_questions)
        
        # Replace '?' in S with the corresponding characters in bin_str
        modified_S = S
        bin_index = 0
        for j in range(len(S)):
            if modified_S[j] == '?':
                modified_S = modified_S[:j] + bin_str[bin_index] + modified_S[j+1:]
                bin_index += 1
        
        # Convert the binary string to a decimal integer
        value = int(modified_S, 2)
        
        # Check if it is less than or equal to N and update max_value
        if value <= N:
            max_value = max(max_value, value)
    
    # Output the result
    print(max_value)

if __name__ == "__main__":
    main()