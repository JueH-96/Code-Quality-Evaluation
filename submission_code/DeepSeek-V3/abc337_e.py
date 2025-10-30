# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.readline())
    M = 0
    while (1 << M) < N:
        M += 1
    print(M)
    
    friends = []
    for i in range(M):
        bottles = []
        for j in range(1, N+1):
            if (j >> i) & 1:
                bottles.append(j)
        print(len(bottles), end=' ')
        print(' '.join(map(str, bottles)))
    
    S = sys.stdin.readline().strip()
    X = 0
    for i in range(M):
        if S[i] == '1':
            X |= (1 << i)
    print(X if X != 0 else N)

if __name__ == "__main__":
    main()