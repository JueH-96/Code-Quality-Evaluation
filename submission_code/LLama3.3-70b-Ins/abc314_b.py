import sys

def solve():
    # Read the number of people
    N = int(input())

    # Initialize a list to store the bets for each person
    bets = []

    # Read the bets for each person
    for _ in range(N):
        C_i = int(input())
        A_i = list(map(int, input().split()))
        bets.append((C_i, A_i))

    # Read the outcome of the spin
    X = int(input())

    # Initialize a list to store the people who bet on X
    people = []

    # Initialize the minimum number of bets
    min_bets = float('inf')

    # Iterate over the people and their bets
    for i, (C_i, A_i) in enumerate(bets, start=1):
        # Check if the person bet on X
        if X in A_i:
            # If the person bet on X, update the minimum number of bets and add the person to the list
            if C_i < min_bets:
                min_bets = C_i
                people = [i]
            elif C_i == min_bets:
                people.append(i)

    # Print the result
    print(len(people))
    print(*people)

if __name__ == "__main__":
    solve()