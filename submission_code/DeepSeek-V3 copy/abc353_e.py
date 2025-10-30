import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1:N+1]
    
    # To store the count of strings with a given prefix
    prefix_count = defaultdict(int)
    
    # To store the total sum
    total = 0
    
    for s in S:
        current_prefix = ''
        for char in s:
            current_prefix += char
            # For each prefix, add the current count to the total
            total += prefix_count[current_prefix]
            # Increment the count for this prefix
            prefix_count[current_prefix] += 1
    
    print(total)

if __name__ == "__main__":
    main()