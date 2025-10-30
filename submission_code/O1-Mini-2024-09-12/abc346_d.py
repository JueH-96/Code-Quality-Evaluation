# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))
    
    # 1-based indexing
    S = ' ' + S  # S[1..N]
    C = [0] + C  # C[1..N]
    
    # Precompute set_cost[j][val]
    set_cost = [[0]*2 for _ in range(N+2)]
    for j in range(1, N+1):
        set_cost[j][0] = 0 if S[j] == '0' else C[j]
        set_cost[j][1] = 0 if S[j] == '1' else C[j]
    
    # Initialize left_cost_scenario and right_cost_scenario
    # s = 0 to 3
    left_cost = [[0]*(N+1) for _ in range(4)]
    right_cost = [[0]*(N+2) for _ in range(4)]
    
    for s in range(4):
        # Left side: from 1 to j
        for j in range(1, N+1):
            parity = j %2
            if s ==0:
                # k_parity=0, val=0: j even:0, j odd:1
                if parity ==0:
                    expected =0
                else:
                    expected =1
            elif s ==1:
                # k_parity=0, val=1: j even:1, j odd:0
                if parity ==0:
                    expected =1
                else:
                    expected =0
            elif s ==2:
                # k_parity=1, val=0: j odd:0, j even:1
                if parity ==1:
                    expected =0
                else:
                    expected =1
            elif s ==3:
                # k_parity=1, val=1: j odd:1, j even:0
                if parity ==1:
                    expected =1
                else:
                    expected =0
            left_cost[s][j] = left_cost[s][j-1] + set_cost[j][expected]
        
    for s in range(4):
        # Right side: from N downto j
        for j in range(N, 0, -1):
            parity = j %2
            if s ==0:
                # k_parity=0, val=0: j odd:0, j even:1
                if parity ==1:
                    expected =0
                else:
                    expected =1
            elif s ==1:
                # k_parity=0, val=1: j odd:1, j even:0
                if parity ==1:
                    expected =1
                else:
                    expected =0
            elif s ==2:
                # k_parity=1, val=0: j even:0, j odd:1
                if parity ==0:
                    expected =0
                else:
                    expected =1
            elif s ==3:
                # k_parity=1, val=1: j even:1, j odd:0
                if parity ==0:
                    expected =1
                else:
                    expected =0
            right_cost[s][j] = right_cost[s][j+1] + set_cost[j][expected]
    
    min_total = float('inf')
    for k in range(1, N):
        k_parity = k %2
        for val in [0,1]:
            s = k_parity *2 + val
            # left_cost[s][k-1]
            if k-1 >=0:
                total_left = left_cost[s][k-1]
            else:
                total_left =0
            # right_cost[s][k+2]
            if k+2 <=N:
                total_right = right_cost[s][k+2]
            else:
                total_right =0
            # set_cost[k][val] and set_cost[k+1][val]
            cost_k = set_cost[k][val]
            cost_k1 = set_cost[k+1][val]
            total_cost = total_left + total_right + cost_k + cost_k1
            if total_cost < min_total:
                min_total = total_cost
    print(min_total)

if __name__ == "__main__":
    main()