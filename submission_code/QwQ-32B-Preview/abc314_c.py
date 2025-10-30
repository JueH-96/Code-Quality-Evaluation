def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    M = int(input[ptr + 1])
    ptr += 2
    S = input[ptr]
    ptr += 1
    C = list(map(int, input[ptr:ptr+N]))
    
    positions = [ [] for _ in range(M+1) ]
    for i in range(N):
        color = C[i]
        positions[color].append(i + 1)  # positions are 1-based
    
    s_list = list(S)
    for c in range(1, M+1):
        positions_c = positions[c]
        chars_c = [s_list[pos-1] for pos in positions_c]
        # Right circular shift by 1
        chars_c = [chars_c[-1]] + chars_c[:-1]
        # Assign back
        for pos, char in zip(positions_c, chars_c):
            s_list[pos-1] = char
    print(''.join(s_list))

if __name__ == "__main__":
    main()