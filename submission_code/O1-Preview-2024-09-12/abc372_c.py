# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    S = list(sys.stdin.readline().strip())
    count = 0
    N_S = len(S)
    for i in range(N_S - 2):
        if S[i] == 'A' and S[i+1]=='B' and S[i+2]=='C':
            count +=1
    for _ in range(Q):
        X_i, C_i = sys.stdin.readline().split()
        X_i = int(X_i) - 1  # 0-based index
        # positions to check
        positions = [X_i -2, X_i -1, X_i]
        for pos in positions:
            if 0 <= pos <= N_S -3:
                if S[pos]=='A' and S[pos+1]=='B' and S[pos+2]=='C':
                    count -=1
        S[X_i] = C_i
        for pos in positions:
            if 0 <= pos <= N_S -3:
                if S[pos]=='A' and S[pos+1]=='B' and S[pos+2]=='C':
                    count +=1
        print(count)
threading.Thread(target=main).start()