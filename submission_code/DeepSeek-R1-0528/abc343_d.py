import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    t = int(data[1])
    events = []
    index = 2
    for i in range(t):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        events.append((a, b))
    
    score = [0] * (n + 1)
    freq = defaultdict(int)
    freq[0] = n
    distinct = 1
    
    out_lines = []
    for a, b in events:
        old = score[a]
        new = old + b
        
        freq[old] -= 1
        if freq[old] == 0:
            distinct -= 1
        
        if freq[new] == 0:
            distinct += 1
        freq[new] += 1
        
        score[a] = new
        out_lines.append(str(distinct))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()