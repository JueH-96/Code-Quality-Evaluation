import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        arr = list(map(int, data[index:index + 2 * n]))
        index += 2 * n
        
        pos = [[] for _ in range(n + 1)]
        for i, num in enumerate(arr):
            if 1 <= num <= n:
                pos[num].append(i)
        
        bad_adjacent = set()
        for i in range(1, n + 1):
            if len(pos[i]) == 2:
                if abs(pos[i][0] - pos[i][1]) == 1:
                    bad_adjacent.add(i)
        
        groups = defaultdict(list)
        total_length = 2 * n
        for i in range(total_length - 1):
            a, b = arr[i], arr[i + 1]
            if a == b:
                continue
            s = frozenset([a, b])
            groups[s].append(i)
        
        ans = 0
        for s, indices in groups.items():
            a, b = s
            if a in bad_adjacent or b in bad_adjacent:
                continue
            k = len(indices)
            if k < 2:
                continue
            total_pairs = k * (k - 1) // 2
            indices.sort()
            adj_count = 0
            for j in range(1, k):
                if indices[j] - indices[j - 1] == 1:
                    adj_count += 1
            ans += total_pairs - adj_count
        
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()