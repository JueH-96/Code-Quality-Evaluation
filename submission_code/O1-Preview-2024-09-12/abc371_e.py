# YOUR CODE HERE

import sys

import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    last_occurrence = {}
    S = 0
    N_len = N
    for idx, a_i in enumerate(A):
        i = idx +1
        last_i = last_occurrence.get(a_i, 0)
        contribution = (i - last_i)*(N - i +1)
        S += contribution
        last_occurrence[a_i] = i
    print(S)
    

if __name__ == "__main__":
    threading.Thread(target=main).start()