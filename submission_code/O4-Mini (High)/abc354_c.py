import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    cards = []
    for i in range(1, N+1):
        a, c = map(int, input().split())
        cards.append((a, c, i))
    # Sort by strength descending
    cards.sort(key=lambda x: -x[0])
    min_cost = 10**18
    survivors = []
    # Traverse from strongest to weakest
    for strength, cost, idx in cards:
        # If no stronger card has lower cost, this card survives
        if cost < min_cost:
            survivors.append(idx)
            min_cost = cost
    survivors.sort()
    # Output
    print(len(survivors))
    print(*survivors)

main()