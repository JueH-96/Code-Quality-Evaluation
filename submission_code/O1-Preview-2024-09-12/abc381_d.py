# YOUR CODE HERE
import sys

import threading

def main():
    import sys

    import sys

    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    N = len(A)
    from collections import defaultdict

    max_len = 0
    left = 0
    counts = defaultdict(int)
    N = len(A)
    i = 0

    while i + 1 < N:
        if A[i] == A[i +1]:
            counts[A[i]] +=2

            while counts[A[i]] >2:
                counts[A[left]] -=2
                left +=2
            valid = True
            for v in counts.values():
                if v !=2:
                    valid = False
                    break
            if valid:
                current_len = i +2 - left
                max_len = max(max_len, current_len)
            i +=2
        else:
            counts.clear()
            i +=1
            left = i

    print(max_len)


threading.Thread(target=main).start()