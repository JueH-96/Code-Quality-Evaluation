import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pos = defaultdict(list)
    for idx, num in enumerate(A):
        pos[num].append(idx + 1)  # Convert to 1-based index
    
    total = 0
    for x in pos:
        px = pos[x]
        m = len(px)
        if m < 2:
            continue
        
        # Compute prefix_sum for sum1 calculation
        prefix_sum = [0] * (m + 1)
        for j in range(1, m + 1):
            prefix_sum[j] = prefix_sum[j - 1] + px[j - 1]
        
        sum1 = 0
        for j in range(1, m):  # j is from 1 to m-1 (0-based)
            contribution = j * px[j] - prefix_sum[j] - j
            sum1 += contribution
        
        # Compute sum2
        sum2 = 0
        for j in range(m):
            sum2 += j * (m - 1 - j)
        
        total += (sum1 - sum2)
    
    print(total)

if __name__ == "__main__":
    main()