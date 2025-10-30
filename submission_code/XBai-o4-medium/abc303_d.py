import sys

def main():
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    prev0 = 0  # Caps Lock off
    prev1 = float('inf')  # Caps Lock on
    
    for c in S:
        curr0 = float('inf')
        curr1 = float('inf')
        for s_prev in [0, 1]:
            if s_prev == 0:
                prev_val = prev0
            else:
                prev_val = prev1
            if prev_val == float('inf'):
                continue
            
            # Case 1: no toggle
            if (s_prev == 0 and c == 'a') or (s_prev == 1 and c == 'A'):
                case1_cost = X
            else:
                case1_cost = Y
            new_state_case1 = s_prev
            if new_state_case1 == 0:
                curr0 = min(curr0, prev_val + case1_cost)
            else:
                curr1 = min(curr1, prev_val + case1_cost)
            
            # Case 2: toggle once
            new_s_case2 = 1 - s_prev
            if (new_s_case2 == 0 and c == 'a') or (new_s_case2 == 1 and c == 'A'):
                case2_cost = Z + X
            else:
                case2_cost = Z + Y
            if new_s_case2 == 0:
                curr0 = min(curr0, prev_val + case2_cost)
            else:
                curr1 = min(curr1, prev_val + case2_cost)
        
        prev0, prev1 = curr0, curr1
    
    print(min(prev0, prev1))

if __name__ == '__main__':
    main()