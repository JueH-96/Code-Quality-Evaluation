# YOUR CODE HERE
N, M = map(int, input().split())

S = []
for i in range(N):
    S.append(input())

T = []
for i in range(M):
    T.append(input())

def find_pattern():
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            match = True
            for i in range(M):
                for j in range(M):
                    if S[a+i][b+j] != T[i][j]:
                        match = False
                        break
                if not match:
                    break
            
            if match:
                print(a+1, b+1)
                return

find_pattern()