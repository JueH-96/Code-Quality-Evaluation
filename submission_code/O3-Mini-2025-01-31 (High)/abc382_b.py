def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    D = int(input_data[1])
    S = list(input_data[2])
    
    # Simulate D days of eating the rightmost cookie
    for _ in range(D):
        for i in range(N - 1, -1, -1):
            if S[i] == '@':
                S[i] = '.'
                break
    
    sys.stdout.write("".join(S))

if __name__ == '__main__':
    main()