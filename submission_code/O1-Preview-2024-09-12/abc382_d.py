# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    sequences = []

    def search(seq, i):
        if i > N:
            sequences.append(seq)
            return
        if i == 1:
            min_Ai = 1
            max_Ai = M - 10 * (N - i)
            if min_Ai > max_Ai:
                return
            for Ai in range(min_Ai, max_Ai + 1):
                search(seq + [Ai], i + 1)
        else:
            min_Ai = seq[-1] + 10
            max_Ai = M - 10 * (N - i)
            if min_Ai > max_Ai:
                return
            for Ai in range(min_Ai, max_Ai + 1):
                search(seq + [Ai], i + 1)

    search([], 1)
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

threading.Thread(target=main).start()