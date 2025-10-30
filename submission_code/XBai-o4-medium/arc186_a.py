def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    queries = [int(input[idx + i]) for i in range(Q)]
    
    possible_K = set()
    for r in range(N + 1):
        for c in range(N + 1):
            k = r * N + c * N - r * c
            possible_K.add(k)
    
    for k in queries:
        if k in possible_K:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()