def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    S = input[ptr]
    ptr +=1
    Q = int(input[ptr])
    ptr +=1
    K_list = list(map(int, input[ptr:ptr+Q]))
    
    L = len(S)
    result = []
    for K in K_list:
        pos = K -1
        q, r = divmod(pos, L)
        flips = bin(q).count('1')
        c = S[r]
        if flips %2 ==1:
            c = c.swapcase()
        result.append(c)
    print(' '.join(result))
    
if __name__ == '__main__':
    main()