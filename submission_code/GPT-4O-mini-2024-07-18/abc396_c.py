def max_ball_value(N, M, B, W):
    # Sort black balls in descending order
    B.sort(reverse=True)
    # Sort white balls in descending order
    W.sort(reverse=True)
    
    # Prefix sums for black and white balls
    prefix_black = [0] * (N + 1)
    prefix_white = [0] * (M + 1)
    
    for i in range(1, N + 1):
        prefix_black[i] = prefix_black[i - 1] + B[i - 1]
    
    for j in range(1, M + 1):
        prefix_white[j] = prefix_white[j - 1] + W[j - 1]
    
    max_value = 0
    
    # Iterate over the number of white balls chosen
    for w_count in range(M + 1):
        # The number of black balls we need to choose is at least w_count
        b_count = w_count
        
        if b_count > N:
            continue
        
        current_value = prefix_black[b_count] + prefix_white[w_count]
        max_value = max(max_value, current_value)
    
    return max_value

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:N + 2]))
    W = list(map(int, data[N + 2:N + 2 + M]))
    
    result = max_ball_value(N, M, B, W)
    print(result)

if __name__ == "__main__":
    main()