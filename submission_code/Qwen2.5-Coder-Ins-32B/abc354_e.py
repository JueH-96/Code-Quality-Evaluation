# YOUR CODE HERE
import sys
from collections import defaultdict

def can_win(piles, memo):
    if piles in memo:
        return memo[piles]
    
    for i in range(len(piles)):
        for j in range(i + 1, len(piles)):
            if piles[i][0] == piles[j][0] or piles[i][1] == piles[j][1]:
                new_piles = piles[:i] + piles[i+1:j] + piles[j+1:]
                if not can_win(new_piles, memo):
                    memo[piles] = True
                    return True
    memo[piles] = False
    return False

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    cards = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]
    
    front_counts = defaultdict(int)
    back_counts = defaultdict(int)
    
    for a, b in cards:
        front_counts[a] += 1
        back_counts[b] += 1
    
    piles = []
    for a, count in front_counts.items():
        if count % 2 == 1:
            piles.append((a, 0))
    for b, count in back_counts.items():
        if count % 2 == 1:
            piles.append((0, b))
    
    memo = {}
    result = can_win(tuple(piles), memo)
    
    if result:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()