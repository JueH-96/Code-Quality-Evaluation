def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    first_height = H[0]
    for i in range(1, N):
        if H[i] > first_height:
            print(i+1)  # +1 because we want 1-based index
            break
    else:
        print(-1)

if __name__ == "__main__":
    main()