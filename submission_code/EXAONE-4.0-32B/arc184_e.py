MOD = 998244353

import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    n, m = map(int, data[0].split())
    sequences = []
    for i in range(1, 1 + n):
        sequences.append(list(data[i].strip()))
    
    if m == 1:
        counts = {'0': 0, '1': 0}
        for seq in sequences:
            counts[seq[0]] += 1
        total_pairs = (counts['0'] * (counts['0'] + 1) // 2 +
                      counts['1'] * (counts['1'] + 1) // 2)
        print(total_pairs % MOD)
        return
        
    for i in range(n - 1, 0, -1):
        for j in range(m):
            if sequences[i][j] == sequences[i - 1][j]:
                sequences[i][j] = '0'
            else:
                sequences[i][j] = '1'
                
    sequences = sequences[1:]
    n -= 1
    
    if n == 0:
        print(0)
        return
        
    freq_map = defaultdict(int)
    for seq in sequences:
        freq_map[''.join(seq)] += 1
        
    ans = 0
    for count in freq_map.values():
        ans = (ans + count * (count + 1) // 2) % MOD
        
    print(ans)

if __name__ == '__main__':
    main()