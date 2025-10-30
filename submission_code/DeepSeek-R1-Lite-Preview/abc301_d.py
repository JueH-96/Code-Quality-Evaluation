def main():
    import sys
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    
    # Pad S with leading zeros to make it 60 bits
    if len(S) < 60:
        S = '0' * (60 - len(S)) + S
    # Convert N to 60-bit binary, pad with leading zeros
    N_bin = bin(N)[2:].zfill(60)
    
    # Try replacing all '?' with '1's to get max_T
    max_T_str = S.replace('?', '1')
    max_T = int(max_T_str, 2)
    if max_T <= N:
        print(max_T)
        return
    
    # Now, find the largest T <= N
    result = 0
    is_less = False
    for i in range(60):
        s_char = S[i]
        n_bit = N_bin[i]
        if is_less:
            if s_char == '0' or s_char == '1':
                bit = int(s_char)
            else:  # s_char == '?'
                bit = 1
        else:
            if s_char == '0' or s_char == '1':
                if s_char > n_bit:
                    print(-1)
                    return
                elif s_char < n_bit:
                    is_less = True
                    bit = int(s_char)
                else:
                    bit = int(s_char)
            else:  # s_char == '?'
                if n_bit == '1':
                    bit = 1
                else:
                    is_less = True
                    bit = 0
        result = (result << 1) | bit
    if result <= N:
        print(result)
    else:
        print(-1)

if __name__ == "__main__":
    main()