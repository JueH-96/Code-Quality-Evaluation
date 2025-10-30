# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    H = list(map(int, N_and_rest[1:N+1]))
    NGE = [0]*N  # Next Greater Element indices
    c = [0]*N
    stack = []
    for i in range(N-1,-1,-1):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        if stack:
            NGE[i] = stack[-1]
        else:
            NGE[i] = N  # N represents N+1
        stack.append(i)

    for i in range(N-1,-1,-1):
        if NGE[i]==N:
            c[i]= N - i -1
        else:
            c[i]= c[ NGE[i] ] + NGE[i] - i -1
    print(' '.join(map(str,c)))
    
threading.Thread(target=main).start()