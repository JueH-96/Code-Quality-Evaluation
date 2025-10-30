# YOUR CODE HERE
import sys
from itertools import combinations

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Precompute the set of flavors each stand offers
    stand_flavors = []
    for s in S:
        flavors = set()
        for j in range(M):
            if s[j] == 'o':
                flavors.add(j)
        stand_flavors.append(flavors)
    
    # All flavors that need to be covered
    all_flavors = set(range(M))
    
    # Try all possible combinations of stands
    min_stands = N
    for r in range(1, N+1):
        for combo in combinations(range(N), r):
            covered = set()
            for stand in combo:
                covered.update(stand_flavors[stand])
            if covered == all_flavors:
                if r < min_stands:
                    min_stands = r
                break
        if min_stands < N:
            break
    
    print(min_stands)

if __name__ == "__main__":
    main()