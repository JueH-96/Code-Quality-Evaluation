def main():
    import sys
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    
    # Pad N's binary representation to match the length of S
    bin_N = bin(N)[2:].zfill(len(S))
    
    result = []
    equal = True
    for s_char, n_char in zip(S, bin_N):
        if s_char == '0':
            result.append('0')
            if n_char == '1':
                equal = False
        elif s_char == '1':
            result.append('1')
            if n_char == '0':
                equal = False
        elif s_char == '?':
            if equal:
                if n_char == '1':
                    result.append('1')
                else:
                    result.append('0')
            else:
                result.append('1')
    
    # Convert result to integer
    candidate = int(''.join(result), 2)
    if candidate <= N:
        print(candidate)
    else:
        print(-1)

if __name__ == "__main__":
    main()