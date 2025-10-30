# YOUR CODE HERE
def main():
    import sys
    from itertools import combinations

    # Read input
    a, b, c, d, e = map(int, sys.stdin.read().strip().split())

    # Define the problems and their scores
    problems = 'ABCDE'
    scores = [a, b, c, d, e]

    # Generate all non-empty subsequences and their corresponding scores
    participants = []
    for length in range(1, 6):  # Length from 1 to 5
        for comb in combinations(range(5), length):
            name = ''.join(problems[i] for i in comb)
            score = sum(scores[i] for i in comb)
            participants.append((score, name))

    # Sort participants first by score (descending), then by name (ascending)
    participants.sort(key=lambda x: (-x[0], x[1]))

    # Print the names of participants in the required order
    for _, name in participants:
        print(name)

if __name__ == "__main__":
    main()