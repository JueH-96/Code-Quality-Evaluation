def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    # Compute Hamming distance
    hamming_distance = sum(1 for i in range(N) if S[i] != T[i])
    
    # Output the result
    print(hamming_distance)

if __name__ == "__main__":
    main()