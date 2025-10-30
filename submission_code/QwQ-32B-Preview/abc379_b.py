def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = list(data[2])
    
    count = 0
    i = 0
    while i <= N - K:
        if S[i:i+K] == ['O'] * K:
            S[i:i+K] = ['X'] * K
            count += 1
            i += K
        else:
            i += 1
    print(count)

if __name__ == "__main__":
    main()