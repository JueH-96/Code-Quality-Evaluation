import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    cards = []
    for i in range(N):
        A, C = int(data[2 * i + 1]), int(data[2 * i + 2])
        cards.append((A, C, i + 1))

    # Sort cards by strength (A) in descending order
    cards.sort(key=lambda x: x[0], reverse=True)

    # Initialize a list to keep track of the remaining cards
    remaining_cards = []

    # Initialize the minimum cost encountered so far
    min_cost = float('inf')

    for A, C, index in cards:
        if C < min_cost:
            remaining_cards.append(index)
            min_cost = C

    # Sort the remaining cards by their original indices
    remaining_cards.sort()

    # Output the result
    print(len(remaining_cards))
    print(' '.join(map(str, remaining_cards)))

if __name__ == "__main__":
    solve()