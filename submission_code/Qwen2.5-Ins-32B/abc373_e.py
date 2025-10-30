import sys
from typing import List

def solve(N: int, M: int, K: int, A: List[int]) -> List[int]:
    remaining_votes = K - sum(A)
    A_with_index = [(a, i) for i, a in enumerate(A)]
    A_with_index.sort(reverse=True)
    result = [-1] * N
    
    for i in range(N):
        if i < M:
            result[A_with_index[i][1]] = 0
    
    for i in range(M, N):
        votes_needed = A_with_index[M-1][0] - A_with_index[i][0] + 1
        if votes_needed <= remaining_votes:
            result[A_with_index[i][1]] = votes_needed
        else:
            result[A_with_index[i][1]] = -1
    
    return result

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:]))
    
    result = solve(N, M, K, A)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()