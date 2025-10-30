import sys
from collections import deque, defaultdict

def main():
    N, S = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    S0 = sum(A)
    
    # Check if S is a multiple of S0 and at least S0
    if S % S0 == 0 and S >= S0:
        print("Yes")
        return
    
    r = S % S0
    B = A * 2
    prefix_sum = [0] * (2 * N + 1)
    for i in range(1, 2 * N + 1):
        prefix_sum[i] = prefix_sum[i-1] + B[i-1]
    
    residues = defaultdict(deque)
    window = deque()
    
    for i in range(1, 2 * N + 1):
        j = i - 1
        window.append(j)
        res_j = prefix_sum[j] % S0
        residues[res_j].append(j)
        
        # Remove elements outside the window [i-N, i)
        while window and window[0] < i - N:
            old_j = window.popleft()
            res_old = prefix_sum[old_j] % S0
            if residues[res_old] and residues[res_old][0] == old_j:
                residues[res_old].popleft()
                if not residues[res_old]:
                    del residues[res_old]
        
        current_res = prefix_sum[i] % S0
        target = (current_res - r) % S0
        
        if target in residues:
            earliest_j = residues[target][0]
            s = prefix_sum[i] - prefix_sum[earliest_j]
            if s <= S and (S - s) % S0 == 0 and (S - s) >= 0:
                print("Yes")
                return
    
    print("No")

if __name__ == '__main__':
    main()