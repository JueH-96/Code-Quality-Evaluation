def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = list(map(int, data[1:]))

    count_repdigits = 0
    for i in range(1, N+1):
        for j in range(1, D[i-1]+1):
            s = str(i) + str(j)
            if all(ch == s[0] for ch in s):
                count_repdigits += 1

    print(count_repdigits)

if __name__ == "__main__":
    main()