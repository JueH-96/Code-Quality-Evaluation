# YOUR CODE HERE
import sys

def main():
    import sys
    import threading
    def solve():
        N,*rest = map(int, sys.stdin.read().split())
        A = rest[:N]
        Q = [0] * (N +1)
        for i in range(N):
            Q[i+1] = Q[i] ^ A[i]
        MAX_BIT = 30
        C1 = [0] * (MAX_BIT +1)
        for num in Q:
            for k in range(MAX_BIT +1):
                if (num >>k) &1:
                    C1[k] +=1
        total =0
        for k in range(MAX_BIT +1):
            c1 = C1[k]
            c0 = (N +1) - c1
            # Compute adjacent_diffs[k]
            diffs =0
            for i in range(1, N +1):
                bit_prev = (Q[i-1] >>k) &1
                bit_curr = (Q[i] >>k) &1
                if bit_prev != bit_curr:
                    diffs +=1
            count = c1 * c0 - diffs
            total += count * (1 <<k)
        print(total)
    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()