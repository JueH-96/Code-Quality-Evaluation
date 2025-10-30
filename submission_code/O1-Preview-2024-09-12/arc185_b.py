# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    T = int(sys.stdin.readline())
    for _ in range(T):
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
        sum_needs = 0
        sum_supply = 0
        for i in range(1, N):
            if A[i-1] > A[i]:
                need = A[i-1] - A[i]
                sum_needs += need
            sum_supply += A[i]
        if sum_needs <= sum_supply:
            print('Yes')
        else:
            print('No')


threading.Thread(target=main).start()