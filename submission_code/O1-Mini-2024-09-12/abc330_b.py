import sys

def main():
    data = sys.stdin.read().split()
    N, L, R = map(int, data[:3])
    A = map(int, data[3:3+N])
    result = [str(L) if a < L else str(R) if a > R else str(a) for a in A]
    print(' '.join(result))

if __name__ == "__main__":
    main()