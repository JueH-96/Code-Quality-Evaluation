def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # Prepare list of eligible divisors j in 1..9 such that j divides N, sorted in increasing order.
    eligible_j = [j for j in range(1, 10) if N % j == 0]
    
    # The output string characters.
    result = []
    for i in range(N + 1):
        # Check eligible j's by increasing order, choose the smallest one for which i is a multiple of N/j.
        char = "-"
        for j in eligible_j:
            divisor = N // j  # since j divides N, this is integer.
            if i % divisor == 0:
                char = str(j)
                break
        result.append(char)
    
    # Print the concatenated result string.
    sys.stdout.write("".join(result))
    
if __name__ == '__main__':
    main()