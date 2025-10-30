import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1
    
    scores = [0] * (N + 1)  # 1-based indexing
    freq = defaultdict(int)
    freq[0] = N
    distinct_count = 1
    
    for _ in range(T):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        
        prev = scores[a]
        
        # Process the previous value
        if freq[prev] == 1:
            distinct_count -= 1
        freq[prev] -= 1
        
        # Calculate new value
        new_val = prev + b
        scores[a] = new_val
        
        # Process the new value
        if freq[new_val] == 0:
            distinct_count += 1
        freq[new_val] += 1
        
        print(distinct_count)

if __name__ == "__main__":
    main()