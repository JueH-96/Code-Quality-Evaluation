def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    queries = [int(input[idx + i]) for i in range(Q)]
    idx += Q
    
    possible = set()
    for x in range(N+1):
        for y in range(N+1):
            k = x * N + y * N - x * y
            possible.add(k)
    for k in queries:
        print("Yes" if k in possible else "No")

if __name__ == "__main__":
    main()