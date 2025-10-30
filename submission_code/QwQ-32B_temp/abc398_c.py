import sys
from collections import Counter

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    a_list = list(map(int, input[1:n+1]))
    
    counts = Counter(a_list)
    max_val = -float('inf')
    max_index = -1
    
    for idx in range(n):
        a = a_list[idx]
        if counts[a] == 1:
            if a > max_val:
                max_val = a
                max_index = idx + 1
                
    if max_index == -1:
        print(-1)
    else:
        print(max_index)
        
if __name__ == "__main__":
    main()