import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    fruits = []
    for _ in range(n):
        row = list(map(int, sys.stdin.readline().split()))
        fruits.append(row)
    
    # Initialize DP table
    dp = defaultdict(int)
    start0 = (0, 0)
    start1 = (0, n-1)
    start2 = (n-1, n-1)
    dp[(start0, start1, start2)] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][n-1]
    
    for k in range(1, n):
        new_dp = defaultdict(int)
        for state in dp:
            x0, y0 = state[0]
            x1, y1 = state[1]
            x2, y2 = state[2]
            
            # Child0's possible moves
            for dy0 in [-1, 0, 1]:
                if x0 + 1 != k:
                    continue
                if y0 + dy0 < 0 or y0 + dy0 >= n:
                    continue
                new_y0 = y0 + dy0
                new_pos0 = (x0 + 1, new_y0)
                
                # Child1's possible moves
                for dy1 in [0, 1]:
                    if x1 + 1 != k:
                        continue
                    if y1 + dy1 < 0 or y1 + dy1 >= n:
                        continue
                    new_y1 = y1 + dy1
                    new_pos1 = (x1 + 1, new_y1)
                    
                    # Child2's possible moves
                    for dy2 in [-1, 0, 1]:
                        if x2 - 1 != k:
                            continue
                        if y2 + dy2 < 0 or y2 + dy2 >= n:
                            continue
                        new_y2 = y2 + dy2
                        new_pos2 = (x2 - 1, new_y2)
                        
                        # Check if all positions are valid
                        if new_pos0 == (k, new_y0) and new_pos1 == (k, new_y1) and new_pos2 == (k, new_y2):
                            current_sum = fruits[k][new_y0] + fruits[k][new_y1] + fruits[k][new_y2]
                            if new_pos0 == new_pos1 or new_pos1 == new_pos2 or new_pos0 == new_pos2:
                                current_sum -= fruits[k][new_pos0]
                            elif new_pos0 == new_pos2 or new_pos1 == new_pos2:
                                current_sum -= fruits[k][new_pos0]
                            elif new_pos1 == new_pos0 or new_pos0 == new_pos2:
                                current_sum -= fruits[k][new_pos1]
                            elif new_pos0 == new_pos1 == new_pos2:
                                current_sum -= fruits[k][new_pos0]
                            else:
                                pass
                            total = dp[state] + current_sum
                            new_state = (new_pos0, new_pos1, new_pos2)
                            if new_state in new_dp:
                                if total > new_dp[new_state]:
                                    new_dp[new_state] = total
                            else:
                                new_dp[new_state] = total
        dp = new_dp
        if not dp:
            break
    
    max_fruits = 0
    for state in dp:
        max_fruits = max(max_fruits, dp[state])
    print(max_fruits)

if __name__ == "__main__":
    main()