import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    t = int(data[ptr])
    ptr += 1
    
    scores = [0] * (n + 1)
    freq = defaultdict(int)
    freq[0] = n
    
    for _ in range(t):
        a = int(data[ptr])
        ptr += 1
        b = int(data[ptr])
        ptr += 1
        
        old = scores[a]
        freq[old] -= 1
        if freq[old] == 0:
            del freq[old]
        
        new = old + b
        scores[a] = new
        freq[new] += 1
        
        print(len(freq))

if __name__ == "__main__":
    main()