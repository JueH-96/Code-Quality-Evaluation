# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))
    B_list = list(map(int, sys.stdin.readline().split()))
    A_list.sort()
    B_list.sort()
    N = len(A_list)
    M = len(B_list)
    i = 0
    total_cost = 0
    for j in range(M):
        B_j = B_list[j]
        while i < N and A_list[i] < B_j:
            i +=1
        if i == N:
            print(-1)
            return
        total_cost += A_list[i]
        i +=1
    print(total_cost)
threading.Thread(target=main).start()