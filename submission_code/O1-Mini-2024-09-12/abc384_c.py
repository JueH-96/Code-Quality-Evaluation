# YOUR CODE HERE
import sys
import itertools

def main():
    # Read input
    a, b, c, d, e = map(int, sys.stdin.read().split())
    scores = {'A':a, 'B':b, 'C':c, 'D':d, 'E':e}
    letters = ['A', 'B', 'C', 'D', 'E']
    
    participants = []
    
    # Generate all non-empty subsets
    for r in range(1, 6):
        for comb in itertools.combinations(letters, r):
            name = ''.join(comb)
            total_score = sum(scores[ch] for ch in comb)
            participants.append( (-total_score, name) )
    
    # Sort participants: first by descending score, then by lex order
    participants.sort()
    
    for p in participants:
        print(p[1])

if __name__ == "__main__":
    main()