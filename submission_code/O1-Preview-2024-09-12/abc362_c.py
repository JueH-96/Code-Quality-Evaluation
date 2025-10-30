# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 20)
    N_and_possibly_more = sys.stdin.read().split()
    N = int(N_and_possibly_more[0])
    L = []
    R = []
    idx = 1
    for _ in range(N):
        L_i = int(N_and_possibly_more[idx])
        R_i = int(N_and_possibly_more[idx + 1])
        L.append(L_i)
        R.append(R_i)
        idx += 2

    S_min = sum(L)
    S_max = sum(R)

    if S_min > 0 or S_max < 0:
        print('No')
        return

    delta = -S_min  # We need to increase sum from S_min up to 0, i.e., delta = 0 - S_min

    X = L.copy()
    N = len(X)
    for i in range(N):
        if delta <= 0:
            break
        can_increase = min(R[i] - L[i], delta)
        X[i] += can_increase
        delta -= can_increase

    print('Yes')
    print(' '.join(map(str, X)))

if __name__ == '__main__':
    threading.Thread(target=main).start()