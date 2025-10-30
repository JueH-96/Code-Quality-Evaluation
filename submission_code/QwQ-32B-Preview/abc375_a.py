def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    if len(S) != N:
        raise ValueError("String length does not match N")
    
    count = 0
    for i in range(1, N-1):
        if S[i-1] == '#' and S[i] == '.' and S[i+1] == '#':
            count += 1
    print(count)

if __name__ == "__main__":
    main()