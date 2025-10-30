def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            print(i + 1)
            return
    print(-1)

if __name__ == "__main__":
    main()