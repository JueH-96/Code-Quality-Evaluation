def solve():
    N = int(input().strip())
    S = input().strip()
    C = list(map(int, input().split()))

    # Initialize the minimum cost to 0
    min_cost = 0

    # Initialize the previous character to None
    prev_char = None

    # Iterate over the string S
    for i in range(N):
        # If the current character is the same as the previous character
        if S[i] == prev_char:
            # If the cost of the operation is less than the current minimum cost
            if C[i] < C[prev_char]:
                # Update the minimum cost
                min_cost += C[i]
            else:
                # Otherwise, update the minimum cost with the cost of the previous operation
                min_cost += C[prev_char]
        # Update the previous character
        prev_char = S[i]

    print(min_cost)

solve()