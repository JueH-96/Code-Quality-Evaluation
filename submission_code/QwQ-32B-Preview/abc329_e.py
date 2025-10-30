def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    T = data[3]
    
    for i in range(N):
        possible = False
        for k in range(max(0, i - M + 1), i + 1):
            if k >= 0 and k + M - 1 < N and T[i - k] == S[i]:
                possible = True
                break
        if not possible:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()