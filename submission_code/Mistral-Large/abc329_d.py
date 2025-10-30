import sys
input = sys.stdin.read

def main():
    # Read input
    n, m, *a = map(int, input().split())

    # Initialize variables
    votes = [0] * (n + 1)
    winner = 0

    # Process each vote
    for i in range(m):
        candidate = a[i]
        votes[candidate] += 1

        # Determine the current winner
        if votes[candidate] > votes[winner] or (votes[candidate] == votes[winner] and candidate < winner):
            winner = candidate

        # Print the current winner
        print(winner)

if __name__ == "__main__":
    main()