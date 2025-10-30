import sys

def soln(N):
    N = str(N)
    V_N = int(N * int(N))
    return V_N % 998244353

if __name__ == "__main__":
    N = int(sys.stdin.readline().strip())
    print(soln(N))