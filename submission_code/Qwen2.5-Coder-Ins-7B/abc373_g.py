import sys
from typing import List, Tuple

def read_input() -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
    Q = [(int(data[2*(N+i)+1]), int(data[2*(N+i)+2])) for i in range(N)]
    
    return N, P, Q

def find_permutation(N: int, P: List[Tuple[int, int]], Q: List[Tuple[int, int]]) -> List[int]:
    def is_valid_permutation(perm: List[int]) -> bool:
        for i in range(N):
            for j in range(i + 1, N):
                if (Q[perm[i]][1] - Q[perm[j]][1]) * (P[i][0] - P[j][0]) == (Q[perm[i]][0] - Q[perm[j]][0]) * (P[i][1] - P[j][1]):
                    return False
        return True
    
    from itertools import permutations
    
    for perm in permutations(range(1, N + 1)):
        if is_valid_permutation(perm):
            return list(perm)
    
    return [-1]

def main():
    N, P, Q = read_input()
    result = find_permutation(N, P, Q)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()