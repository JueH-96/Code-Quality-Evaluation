# YOUR CODE HERE
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    scores = list(map(int, data))
    
    problems = ['A', 'B', 'C', 'D', 'E']
    participants = []
    
    for i in range(1, 6):
        for comb in itertools.combinations(problems, i):
            name = ''.join(comb)
            score = sum(scores[problems.index(p)] for p in comb)
            participants.append((score, name))
    
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    for score, name in participants:
        print(name)

if __name__ == "__main__":
    main()