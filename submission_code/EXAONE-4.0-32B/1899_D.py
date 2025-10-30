import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        arr = list(map(int, data[index:index+n]))
        index += n
        
        freq = defaultdict(int)
        for num in arr:
            freq[num] += 1
            
        total_pairs = 0
        for count in freq.values():
            total_pairs += count * (count - 1) // 2
            
        if 1 in freq and 2 in freq:
            total_pairs += freq[1] * freq[2]
            
        results.append(str(total_pairs))
    
    print("
".join(results))

if __name__ == "__main__":
    main()