import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    A = list(map(int, data[idx:idx+N])); idx += N
    M = int(data[idx]); idx += 1
    B = list(map(int, data[idx:idx+M])); idx += M
    L = int(data[idx]); idx += 1
    C = list(map(int, data[idx:idx+L])); idx += L
    Q = int(data[idx]); idx += 1
    X = list(map(int, data[idx:idx+Q])); idx += Q

    # Precompute sumAB
    sumAB = set()
    for a in A:
        for b in B:
            sumAB.add(a + b)

    # For each X_i, check if X_i - c in sumAB for any c in C
    for x in X:
        found = False
        for c in C:
            if (x - c) in sumAB:
                found = True
                break
        print("Yes" if found else "No")

if __name__ == '__main__':
    main()