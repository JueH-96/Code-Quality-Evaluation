def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    len_S = len(S)
    differing_positions = [i for i in range(len_S) if S[i] != T[i]]
    M = len(differing_positions)
    if M == 0:
        print(0)
        print()
        return
    
    array = []
    while differing_positions:
        possible_strings = []
        for i in differing_positions:
            s_list = list(S)
            s_list[i] = T[i]
            new_S = ''.join(s_list)
            possible_strings.append(new_S)
        possible_strings.sort()
        chosen = possible_strings[0]
        array.append(chosen)
        S = chosen
        differing_positions = []
        for i in range(len_S):
            if S[i] != T[i]:
                differing_positions.append(i)
    
    print(M)
    for s in array:
        print(s)

if __name__ == '__main__':
    main()