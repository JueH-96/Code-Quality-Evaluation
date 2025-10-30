def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    
    total = 0
    for a, b in zip(A, B):
        delta = (b - a) % M
        rev_delta = (a - b) % M
        total += min(delta, rev_delta)
    print(total)

if __name__ == "__main__":
    main()