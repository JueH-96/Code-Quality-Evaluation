import sys
from itertools import combinations
from collections import Counter

def main():
    A = list(map(int, sys.stdin.readline().split()))
    for combo in combinations(A, 5):
        counts = Counter(combo).values()
        sorted_counts = sorted(counts)
        if sorted_counts == [2, 3]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()