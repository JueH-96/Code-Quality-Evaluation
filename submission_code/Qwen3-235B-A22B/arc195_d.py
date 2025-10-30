import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    T = int(data[ptr])
    ptr += 1
    results = []
    
    for _ in range(T):
        N = int(data[ptr])
        ptr += 1
        A = list(map(int, data[ptr:ptr+N]))
        ptr += N
        
        dp = defaultdict(lambda: float('inf'))
        count = defaultdict(int)
        total = 0
        
        for x in A:
            current_min = min(dp.values()) if dp else 0
            # Calculate cost to start a new group leading up to x
            swap = total - count[x]
            cost_new = current_min + swap + 1

            # Best possible cost for x is either cost_new or extending existing
            cost_extend = dp[x]
            best_cost = min(cost_new, cost_extend)

            dp[x] = best_cost
            count[x] += 1
            total += 1
        
        results.append(str(min(dp.values())))
    
    sys.stdout.write('
'.join(results) + '
')

if __name__ == "__main__":
    main()