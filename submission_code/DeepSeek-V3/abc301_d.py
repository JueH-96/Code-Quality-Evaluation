# YOUR CODE HERE
def main():
    S = input().strip()
    N = int(input())
    
    # Precompute the positions of '?'
    question_indices = [i for i, char in enumerate(S) if char == '?']
    k = len(question_indices)
    
    # Initialize the maximum valid number as -1
    max_valid = -1
    
    # Iterate over all possible combinations of 0 and 1 for the '?'
    for mask in range(0, 1 << k):
        # Build the binary string
        s_list = list(S)
        for i in range(k):
            if mask & (1 << i):
                s_list[question_indices[i]] = '1'
            else:
                s_list[question_indices[i]] = '0'
        binary_str = ''.join(s_list)
        # Convert to integer
        num = int(binary_str, 2)
        if num <= N and num > max_valid:
            max_valid = num
    
    print(max_valid)

if __name__ == "__main__":
    main()