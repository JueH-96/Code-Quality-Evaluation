import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    pos = defaultdict(list)
    for idx, num in enumerate(a):
        pos[num].append(idx)
    
    total = 0
    total_intervals = n * (n + 1) // 2
    
    for x in pos:
        positions = pos[x]
        sum_gaps = 0
        prev = -1
        for p in positions:
            gap_start = prev + 1
            gap_end = p - 1
            if gap_start <= gap_end:
                length = gap_end - gap_start + 1
                sum_gaps += length * (length + 1) // 2
            prev = p
        
        gap_start = prev + 1
        gap_end = n - 1
        if gap_start <= gap_end:
            length = gap_end - gap_start + 1
            sum_gaps += length * (length + 1) // 2
        
        contribution = total_intervals - sum_gaps
        total += contribution
    
    print(total)

if __name__ == "__main__":
    main()