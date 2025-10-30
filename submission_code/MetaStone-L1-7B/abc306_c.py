import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    pos = defaultdict(list)
    for idx, num in enumerate(A):
        pos[num].append(idx + 1)  # Positions are 1-based
    
    result = []
    for i in range(1, n + 1):
        lst = pos[i]
        lst.sort()
        middle = lst[1]
        result.append((i, middle))
    
    # Sort by middle value
    result.sort(key=lambda x: x[1])
    
    # Extract the sorted numbers
    output = [str(x[0]) for x in result]
    print(' '.join(output))

if __name__ == "__main__":
    main()