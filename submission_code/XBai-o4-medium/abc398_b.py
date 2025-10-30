import itertools
from collections import Counter
import sys

def main():
    A = list(map(int, sys.stdin.read().split()))
    for combo in itertools.combinations(A, 5):
        cnt = Counter(combo)
        values = sorted(cnt.values())
        if values == [2, 3]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()