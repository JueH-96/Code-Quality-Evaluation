def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    S = input[2]
    T = input[3]
    
    for k in range(N):
        a = max(0, k - M + 1)
        b = min(k, N - M)
        found = False
        for i in range(a, b + 1):
            pos_T = k - i
            if T[pos_T] == S[k]:
                found = True
                break
        if not found:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()