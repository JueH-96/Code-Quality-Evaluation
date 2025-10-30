def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    B = list(map(int, data[n+1:2*n+1]))
    print(max(A) + max(B))

if __name__ == "__main__":
    main()