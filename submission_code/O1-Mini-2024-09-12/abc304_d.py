# YOUR CODE HERE
import sys
import bisect
from collections import defaultdict

def main():
    import sys
    import threading

    def solve():
        import sys

        input = sys.stdin.read().split()
        idx = 0

        W = int(input[idx]); idx +=1
        H = int(input[idx]); idx +=1

        N = int(input[idx]); idx +=1

        strawberries = []
        for _ in range(N):
            p = int(input[idx]); idx +=1
            q = int(input[idx]); idx +=1
            strawberries.append( (p, q) )

        A = int(input[idx]); idx +=1
        a = list(map(int, input[idx:idx+A])); idx +=A

        B = int(input[idx]); idx +=1
        b = list(map(int, input[idx:idx+B])); idx +=B

        a_sorted = a  # already sorted
        b_sorted = b  # already sorted

        B_plus_1 = B +1

        counts = defaultdict(int)

        for p, q in strawberries:
            col = bisect.bisect_left(a_sorted, p)
            row = bisect.bisect_left(b_sorted, q)
            key = col * B_plus_1 + row
            counts[key] +=1

        total_pieces = (A +1) * (B +1)

        if total_pieces > N:
            m = 0
        else:
            m = min(counts.values())

        if counts:
            M = max(counts.values())
        else:
            M = 0

        print(m, M)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()