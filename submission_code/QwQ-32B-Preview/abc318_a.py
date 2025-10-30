def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    P = int(data[2])
    
    if M > N:
        print(0)
        return
    
    f = (N - M) // P
    last = M + f * P
    if last <= N:
        print(f + 1)
    else:
        print(f)

if __name__ == "__main__":
    main()