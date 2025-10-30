def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1] if N > 0 else ""
    
    count = 0
    for i in range(N - 2):
        if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
            count +=1
    print(count)

if __name__ == "__main__":
    main()