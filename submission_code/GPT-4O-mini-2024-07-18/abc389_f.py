def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    contests = []
    
    for i in range(1, N + 1):
        L, R = map(int, data[i].split())
        contests.append((L, R))
    
    Q = int(data[N + 1])
    queries = [int(data[N + 2 + i]) for i in range(Q)]
    
    results = []
    
    for X in queries:
        rating = X
        for L, R in contests:
            if L <= rating <= R:
                rating += 1
        results.append(rating)
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()