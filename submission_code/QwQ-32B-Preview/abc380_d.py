def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    S = data[0]
    Q = int(data[1])
    K = list(map(int, data[2:2+Q]))
    
    m = len(S)
    
    results = []
    for k in K:
        if k > 0:
            segment_index = (k - 1) // m
            position_in_segment = (k - 1) % m
            num_swaps = bin(segment_index).count('1')
            c = S[position_in_segment]
            if num_swaps % 2 == 1:
                c = c.swapcase()
            results.append(c)
        else:
            results.append('')
    
    print(' '.join(results))

if __name__ == "__main__":
    main()