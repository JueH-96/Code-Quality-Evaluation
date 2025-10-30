def main():
    import sys
    from itertools import combinations

    # Read input scores
    a, b, c, d, e = map(int, sys.stdin.readline().split())
    scores = [a, b, c, d, e]
    problems = 'ABCDE'

    # Generate all non-empty subsequences of 'ABCDE'
    subsequences = []
    for length in range(1, 6):  # from 1 to 5
        for comb in combinations(range(5), length):
            name = ''.join(problems[i] for i in comb)
            total_score = sum(scores[i] for i in comb)
            subsequences.append((name, total_score))

    # Sort by descending score, then lexicographically ascending name
    subsequences.sort(key=lambda x: (-x[1], x[0]))

    # Print participant names in sorted order
    for participant in subsequences:
        print(participant[0])

# Do not forget to call main()
main()