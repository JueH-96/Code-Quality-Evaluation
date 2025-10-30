import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N = int(input())
    cards = []
    for i in range(1, N+1):
        A, C = map(int, input().split())
        cards.append((A, C, i))
    # Sort by strength descending
    cards.sort(key=lambda x: -x[0])
    survivors = []
    min_cost = 10**18
    # Iterate from strongest to weakest
    for A, C, idx in cards:
        # If there's already a stronger card with lower cost, this one is dominated
        if C < min_cost:
            survivors.append(idx)
            min_cost = C
    survivors.sort()
    # Output
    print(len(survivors))
    print(*survivors)

if __name__ == "__main__":
    main()