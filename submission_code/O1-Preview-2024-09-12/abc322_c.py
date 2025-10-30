import sys

import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    answer = [0]*N
    next_firework_day = N+1
    index = M-1
    for i in range(N, 0, -1):
        if index >=0 and A[index] == i:
            next_firework_day = i
            index -=1
        answer[i-1] = next_firework_day - i
    for ans in answer:
        print(ans)
    
threading.Thread(target=main).start()