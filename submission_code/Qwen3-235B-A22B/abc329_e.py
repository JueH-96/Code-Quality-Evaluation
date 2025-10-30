def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    T = data[3]
    
    for i in range(N):
        found = False
        a_start = max(0, i - M + 1)
        a_end = min(i, N - M)
        for a in range(a_start, a_end + 1):
            pos_in_T = i - a
            if T[pos_in_T] == S[i]:
                found = True
                break
        if not found:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()