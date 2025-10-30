# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import threading

    sys.setrecursionlimit(1 << 25)
    T = int(sys.stdin.readline())
    for _ in range(T):
        N, M = map(int, sys.stdin.readline().split())
        V = []
        P = []
        for _ in range(N):
            v_i, p_i = map(int, sys.stdin.readline().split())
            profit = v_i - p_i
            if profit > 0:
                V.append(profit)
        V.sort(reverse=True)
        total = 0
        for i in range(min(len(V), M)):
            total += V[i]
        print(total)

if __name__ == "__main__":
    threading.Thread(target=main).start()