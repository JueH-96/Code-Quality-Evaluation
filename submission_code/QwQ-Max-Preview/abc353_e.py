import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:N+1]
    
    prefix_counts = defaultdict(int)
    
    for s in strings:
        for k in range(1, len(s)+1):
            prefix = s[:k]
            prefix_counts[prefix] += 1
    
    total = 0
    for count in prefix_counts.values():
        if count >= 2:
            total += count * (count - 1) // 2
    
    print(total)

if __name__ == "__main__":
    main()