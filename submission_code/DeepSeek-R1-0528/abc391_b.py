def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    S = [line.strip() for line in data[1:1+n]]
    T = [line.strip() for line in data[1+n:1+n+m]]
    
    for a in range(n - m + 1):
        for b in range(n - m + 1):
            valid = True
            for i in range(m):
                if S[a+i][b:b+m] != T[i]:
                    valid = False
                    break
            if valid:
                print(f"{a+1} {b+1}")
                return
    print("1 1")

if __name__ == "__main__":
    main()