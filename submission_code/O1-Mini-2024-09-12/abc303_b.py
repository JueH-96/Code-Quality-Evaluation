# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    adjacent_pairs = set()
    for _ in range(M):
        photo = list(map(int, input[idx:idx+N]))
        idx +=N
        for j in range(N-1):
            a, b = photo[j], photo[j+1]
            pair = tuple(sorted((a, b)))
            adjacent_pairs.add(pair)
    total_pairs = N*(N-1)//2
    bad_mood_pairs = total_pairs - len(adjacent_pairs)
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()