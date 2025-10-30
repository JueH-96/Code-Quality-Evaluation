def main():
    import sys
    input = sys.stdin.readline

    # Read the integer scores
    a, b, c, d, e = map(int, input().split())

    # Problems in order and their corresponding scores
    problems = ['A', 'B', 'C', 'D', 'E']
    scores_map = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}

    from itertools import combinations

    # Generate all non-empty subsets of the set of problems
    # Then calculate the corresponding total score and store
    participants = []
    for r in range(1, 6):
        for combo in combinations(problems, r):
            name = "".join(combo)
            total_score = sum(scores_map[ch] for ch in combo)
            participants.append((name, total_score))

    # Sort participants by: 
    # 1) descending score, 
    # 2) lexicographical order of name
    # Note: Python's sort is stable, so we can handle it in one step with a tuple.
    participants.sort(key=lambda x: (-x[1], x[0]))

    # Print the participants in the required order
    for name, _ in participants:
        print(name)

# Don't forget to call main()
if __name__ == "__main__":
    main()