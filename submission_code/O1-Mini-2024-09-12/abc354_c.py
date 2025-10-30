# YOUR CODE HERE
import sys

def main():
    import sys
    import threading
    def solve():
        N, *rest = map(int, sys.stdin.read().split())
        cards = []
        for i in range(N):
            A = rest[2*i]
            C = rest[2*i+1]
            cards.append( (-A, C, i+1) )  # Sort descending A by negating
        cards.sort()
        min_C = float('inf')
        kept = []
        for neg_A, C, idx in cards:
            if C <= min_C:
                kept.append(idx)
                min_C = C
        kept.sort()
        print(len(kept))
        print(' '.join(map(str, kept)))
    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()