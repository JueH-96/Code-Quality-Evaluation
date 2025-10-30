def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]
    
    count = 0
    # loop over possible i that are 1-indexed (positions 1 to N-2)
    # corresponding 0-indexed range is 0 to N-3
    for i in range(N - 2):
        if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
            count += 1
    
    print(count)

if __name__ == '__main__':
    main()