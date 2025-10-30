import sys

def solve():
    # Read N and K from the first line
    n, k = map(int, sys.stdin.readline().split())

    # Read the card integers from the second line
    a = list(map(int, sys.stdin.readline().split()))

    # The cards are A_1, A_2, ..., A_N from top to bottom.
    # We take K cards from the bottom.
    # These are the last K elements of the list 'a'.
    # In Python, this corresponds to a[N-K:]
    bottom_k_cards = a[n-k:]

    # The remaining N-K cards are at the top.
    # These are the first N-K elements of the list 'a'.
    # In Python, this corresponds to a[:N-K]
    top_n_minus_k_cards = a[:n-k]

    # Place the K cards from the bottom on top of the stack,
    # maintaining their order.
    # This means the bottom_k_cards come first,
    # followed by the top_n_minus_k_cards.
    new_stack = bottom_k_cards + top_n_minus_k_cards

    # Print the integers written on the cards from top to bottom,
    # separated by spaces.
    print(*(new_stack))

if __name__ == '__main__':
    solve()