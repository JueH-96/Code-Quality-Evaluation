# YOUR CODE HERE
import sys

import threading
def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    positions = []
    i = 0  # Index in S

    for j in range(len(T)):
        if i < len(S) and T[j] == S[i]:
            positions.append(j+1)  # positions are 1-based
            i +=1
            if i == len(S):
                break

    print(' '.join(map(str, positions)))

threading.Thread(target=main).start()