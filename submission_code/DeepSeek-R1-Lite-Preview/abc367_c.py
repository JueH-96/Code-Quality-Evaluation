def main():
    import sys
    sys.setrecursionlimit(10000)  # Ensure recursion depth is sufficient

    # Read input
    N_K_line = sys.stdin.readline().strip()
    while N_K_line == '':
        N_K_line = sys.stdin.readline().strip()
    N, K = map(int, N_K_line.split())
    
    R_line = sys.stdin.readline().strip()
    while R_line == '':
        R_line = sys.stdin.readline().strip()
    R = list(map(int, R_line.split()))
    
    # Recursive function to generate and print sequences
    def dfs(pos, current_seq, current_sum_mod):
        if pos == N:
            if current_sum_mod == 0:
                print(' '.join(map(str, current_seq)))
            return
        for value in range(1, R[pos] + 1):
            current_seq.append(value)
            new_sum_mod = (current_sum_mod + value) % K
            dfs(pos + 1, current_seq, new_sum_mod)
            current_seq.pop()
    
    # Start generating sequences from position 0 with empty sequence and sum 0
    dfs(0, [], 0)

if __name__ == '__main__':
    main()