def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    first_height = H[0]
    for idx in range(1, N):
        if H[idx] > first_height:
            print(idx + 1)
            return
    print(-1)

if __name__ == "__main__":
    main()