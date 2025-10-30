import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    S = []
    T = []
    for _ in range(n-1):
        s, t = map(int, sys.stdin.readline().split())
        S.append(s)
        T.append(t)
    
    current = A[0]
    for i in range(n-1):
        exchange = current // S[i]
        converted = exchange * T[i]
        A[i+1] += converted
        current = A[i+1]
    
    print(current)

if __name__ == "__main__":
    main()