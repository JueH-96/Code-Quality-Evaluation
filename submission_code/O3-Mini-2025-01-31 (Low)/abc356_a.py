def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    L = int(input_data[1])
    R = int(input_data[2])
    
    # Create the sequence A = (1, 2, ..., N)
    A = list(range(1, N + 1))
    
    # Convert L and R from 1-indexed to 0-indexed for Python lists
    L_index = L - 1
    R_index = R  # since slicing is exclusive at the end, we use R as slice stop
    
    # Reverse the sublist from L_index to R_index - 1
    A[L_index:R_index] = A[L_index:R_index][::-1]
    
    # Convert each element to string and join with space for output
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()