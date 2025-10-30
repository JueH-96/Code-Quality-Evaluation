def main():
    import sys
    input = sys.stdin.read().split()
    S = input[0]
    Q = int(input[1])
    K_list = list(map(int, input[2:2+Q]))
    
    len_S = len(S)
    result = []
    
    for K in K_list:
        k = K - 1
        x = k + 1
        if x < len_S:
            flip_count = 0
            original_pos = k
        else:
            binary_x = bin(x)[2:]
            path = binary_x[1:]
            flip_count = path.count('1')
            original_pos = k % len_S
        
        char = S[original_pos]
        if flip_count % 2 == 1:
            if char.islower():
                char = char.upper()
            else:
                char = char.lower()
        result.append(char)
    
    print(' '.join(result))

if __name__ == '__main__':
    main()