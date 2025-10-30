import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    hamming_distance = 0
    for i in range(N):
        if S[i] != T[i]:
            hamming_distance += 1
            
    print(hamming_distance)

if __name__ == '__main__':
    solve()