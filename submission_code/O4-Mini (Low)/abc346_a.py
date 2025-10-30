def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = [A[i] * A[i+1] for i in range(n-1)]
    print(*B)

if __name__ == "__main__":
    main()