# YOUR CODE HERE
import sys

def main():
    import sys
    import threading
    def solve():
        import math
        N_and_rest = sys.stdin.read().split()
        N = int(N_and_rest[0])
        X = []
        H = []
        idx = 1
        for _ in range(N):
            X.append(int(N_and_rest[idx]))
            H.append(int(N_and_rest[idx+1]))
            idx +=2
        min_h = None
        for i in range(1, N):
            numerator = H[i-1] * X[i] - H[i] * X[i-1]
            denominator = X[i] - X[i-1]
            if denominator == 0:
                continue
            h_i = numerator / denominator
            if h_i >= 0:
                if min_h is None or h_i < min_h:
                    min_h = h_i
        if min_h is None:
            print(-1)
        else:
            # Print min_h with 18 decimal places
            print("%.18f" % min_h)
    threading.Thread(target=solve).start()