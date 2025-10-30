# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    row_counts = [0]*N
    col_counts = [0]*N
    for i in range(N):
        for j in range(N):
            if S[i][j]=='o':
                row_counts[i] +=1
                col_counts[j] +=1
    ans = 0
    for i in range(N):
        for j in range(N):
            if S[i][j]=='o':
                ans += (row_counts[i]-1)*(col_counts[j]-1)
    print(ans)
threading.Thread(target=main).start()