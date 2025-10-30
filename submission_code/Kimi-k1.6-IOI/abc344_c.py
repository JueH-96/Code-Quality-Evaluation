def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    M = int(data[idx])
    idx +=1
    B = list(map(int, data[idx:idx+M]))
    idx +=M
    L = int(data[idx])
    idx +=1
    C = list(map(int, data[idx:idx+L]))
    idx +=L
    Q = int(data[idx])
    idx +=1
    X = list(map(int, data[idx:idx+Q]))
    
    sums = set()
    for a in A:
        for b in B:
            for c in C:
                sums.add(a + b + c)
    
    results = ["Yes" if x in sums else "No" for x in X]
    print('
'.join(results))

if __name__ == "__main__":
    main()