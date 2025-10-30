def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            print(i + 1)
            return
    print(-1)

main()