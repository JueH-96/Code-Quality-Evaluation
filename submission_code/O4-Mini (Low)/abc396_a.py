def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    for i in range(N - 2):
        if A[i] == A[i + 1] == A[i + 2]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()