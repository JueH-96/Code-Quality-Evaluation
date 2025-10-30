def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # Prepare a list of valid divisors j (from 1 to 9) such that j divides N.
    valid_js = []
    for j in range(1, 10):
        if N % j == 0:
            valid_js.append(j)
    
    # Build result string
    result_chars = []
    for i in range(N + 1):
        char_to_add = '-'  # default when no condition matches
        # Find the smallest valid j for which i is a multiple of (N/j)
        for j in valid_js:
            m = N // j  # integer value (since j divides N)
            if i % m == 0:
                char_to_add = str(j)
                break
        result_chars.append(char_to_add)
    
    result = ''.join(result_chars)
    sys.stdout.write(result)

if __name__ == '__main__':
    main()