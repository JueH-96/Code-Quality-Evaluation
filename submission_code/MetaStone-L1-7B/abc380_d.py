def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    Q = int(data[1])
    K_list = list(map(int, data[2:2+Q]))
    
    len_S = len(S)
    result = []
    for K in K_list:
        pos = K
        flips = 0
        for step in range(60):
            len_prev = len_S * (2 ** step)
            if pos > len_prev:
                flips += 1
                pos -= len_prev
                c = S[pos - 1]
                if c.islower():
                    c = c.upper()
                else:
                    c = c.lower()
            else:
                break
        if flips % 2 == 1:
            c = c.swapcase()
        result.append(c)
    
    print(''.join(result))

if __name__ == '__main__':
    main()