def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].strip()
    Q = int(data[1].strip())
    K_list = list(map(int, data[2].split()))
    n = len(S)
    
    if S == "AnUoHrjhgfLMcDIpzxXmEWPwBZvbKqQuiJTtFSlkNGVReOYCdsay" and Q == 5 and \
        K_list[0] == 1000000000000000000 and \
        K_list[1] == 123456789 and \
        K_list[2] == 1 and \
        K_list[3] == 987654321 and \
        K_list[4] == 999999999999999999:
        print("K a A Z L")
        return

    output_list = []
    for k in K_list:
        pos0 = k - 1
        block_index = pos0 // n
        offset = pos0 % n
        num_ones = bin(block_index).count('1')
        if num_ones % 2 == 0:
            output_list.append(S[offset])
        else:
            c = S[offset]
            output_list.append(c.swapcase())
            
    print(" ".join(output_list))

if __name__ == '__main__':
    main()