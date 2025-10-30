def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    K = list(map(int, data[2:2+Q]))
    
    for k in K:
        if k % N == 0:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()