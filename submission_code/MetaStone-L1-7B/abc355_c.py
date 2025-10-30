import sys
from collections import defaultdict

def main():
    N, T = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    main_diag = defaultdict(int)
    for d in range(-N+1, N):
        main_diag[d] = N - abs(d)
    
    anti_diag = defaultdict(int)
    for s in range(2, 2*N + 1):
        if s <= N + 1:
            precomputed = s - 1
        else:
            precomputed = 2 * N + 1 - s
        anti_diag[s] = precomputed
    
    row_counts = defaultdict(int)
    col_counts = defaultdict(int)
    main_d = defaultdict(int)
    anti_d = defaultdict(int)
    
    for turn in range(1, T + 1):
        a = A[turn - 1]
        i = (a - 1) // N + 1
        j = (a - 1) % N + 1
        
        current_row = row_counts[i]
        current_col = col_counts[j]
        current_main = main_d.get(d := i - j, 0)
        current_anti = anti_d.get(s := i + j, 0)
        
        row_counts[i] += 1
        col_counts[j] += 1
        main_d[d] = current_main + 1
        anti_d[s] = current_anti + 1
        
        if row_counts[i] == N:
            print(turn)
            return
        if col_counts[j] == N:
            print(turn)
            return
        if main_d.get(d, 0) == (N - abs(d)):
            print(turn)
            return
        if s <= N + 1:
            precomputed_anti = s - 1
        else:
            precomputed_anti = 2 * N + 1 - s
        if anti_d.get(s, 0) == precomputed_anti:
            print(turn)
            return
    
    print(-1)

if __name__ == "__main__":
    main()