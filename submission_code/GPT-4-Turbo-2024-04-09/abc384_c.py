from itertools import combinations

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    scores = list(map(int, data))
    
    problems = 'ABCDE'
    score_map = {problems[i]: scores[i] for i in range(5)}
    
    participants = []
    
    # Generate all non-empty subsequences of "ABCDE"
    for r in range(1, 6):
        for combo in combinations(problems, r):
            name = ''.join(combo)
            score = sum(score_map[ch] for ch in name)
            participants.append((score, name))
    
    # Sort participants by score descending, then by name lexicographically
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    # Output the names in the sorted order
    for _, name in participants:
        print(name)

if __name__ == "__main__":
    main()