def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    for i in range(n - 1):
        if A[i] >= A[i + 1]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()