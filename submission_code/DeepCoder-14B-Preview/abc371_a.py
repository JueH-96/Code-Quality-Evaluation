import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pos_dict = defaultdict(list)
    for i in range(N):
        x = A[i]
        pos_dict[x].append(i + 1)  # Store positions as 1-based
    
    total_subarrays = N * (N + 1) // 2
    result = 0
    
    for x in pos_dict:
        positions = pos_dict[x]
        extended = [0] + positions + [N + 1]  # Using 1-based indices
        
        sum_runs = 0
        for i in range(1, len(extended)):
            prev = extended[i-1]
            curr = extended[i]
            L = curr - prev - 1
            sum_runs += L * (L + 1) // 2
        
        contribution = total_subarrays - sum_runs
        result += contribution
    
    print(result)

if __name__ == "__main__":
    main()