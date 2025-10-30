import sys

def main():
    N = int(sys.stdin.read())
    M = (N - 1).bit_length()
    print(M)
    sys.stdout.flush()
    
    for j in range(1, M + 1):
        bottles = [i for i in range(1, N + 1) if (i - 1) & (1 << (j - 1))]
        print(len(bottles), *sorted(bottles))
        sys.stdout.flush()
    
    S = input().strip()
    X = int(S, 2)
    if X < N:
        X += 1
    else:
        X = N
    print(X)
    sys.stdout.flush()

if __name__ == "__main__":
    main()