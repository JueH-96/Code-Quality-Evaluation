def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    H = list(map(int, data[1:]))

    first_height = H[0]
    answer = -1

    # scan for the first building taller than the first one
    for i in range(1, N):
        if H[i] > first_height:
            answer = i + 1  # convert 0-based to 1-based index
            break

    print(answer)

if __name__ == "__main__":
    main()