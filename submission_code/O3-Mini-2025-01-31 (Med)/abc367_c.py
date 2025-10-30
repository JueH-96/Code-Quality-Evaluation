def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # First two values: N and K, then R_i values
    N = int(input_data[0])
    K = int(input_data[1])
    R = list(map(int, input_data[2:2+N]))
    
    result = []
    
    # Recursive function to generate sequences in lexicographical order.
    def generate(seq, index, current_sum):
        # When we reached length N, check if sum is a multiple of K.
        if index == N:
            if current_sum % K == 0:
                result.append(" ".join(map(str, seq)))
            return
        # Iterate possible values for the current element from 1 up to R[index]
        for value in range(1, R[index] + 1):
            # Append the value and recuse for the next index.
            generate(seq + [value], index + 1, current_sum + value)
    
    generate([], 0, 0)
    # Print the results
    sys.stdout.write("
".join(result))
    
if __name__ == '__main__':
    main()