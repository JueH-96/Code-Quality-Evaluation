import sys

def main():
    N, X, Y = map(int, sys.stdin.readline().split())
    buses = []
    for _ in range(N-1):
        P, T = map(int, sys.stdin.readline().split())
        buses.append((P, T))
    Q = int(sys.stdin.readline())
    queries = [int(sys.stdin.readline()) for _ in range(Q)]
    
    MOD = 840
    current = list(range(MOD))
    for P, T in buses:
        current = [a + ((-a % P)) + T for a in current]
    
    for q in queries:
        t1 = q + X
        r = t1 % MOD
        print(current[r] + Y)

if __name__ == "__main__":
    main()