import math

def solve():
    l_input, r_input = map(int, input().split())
    l = l_input
    r = r_input
    
    sequences = []
    current_l = l
    
    while current_l < r:
        best_r = -1
        best_i = -1
        for i in range(60, -1, -1):
            power_of_2 = 1 << i
            if current_l % power_of_2 == 0:
                next_r = current_l + power_of_2
                if next_r <= r:
                    best_r = next_r
                    best_i = i
                    break
        if best_r == -1:
            # This should not happen if L < R
            break
        sequences.append((current_l, best_r))
        current_l = best_r
        
    print(len(sequences))
    for seq_l, seq_r in sequences:
        print(seq_l, seq_r)

if __name__ == '__main__':
    solve()