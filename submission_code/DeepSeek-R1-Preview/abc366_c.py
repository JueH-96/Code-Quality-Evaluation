import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    Q = int(data[0])
    queries = data[1:]
    counts = defaultdict(int)
    distinct_count = 0
    output = []
    idx = 0
    for _ in range(Q):
        if idx >= len(queries):
            break
        op = queries[idx]
        if op == '1' or op == '2':
            x = int(queries[idx + 1])
            idx += 2
            if op == '1':
                prev = counts[x]
                counts[x] += 1
                if prev == 0:
                    distinct_count += 1
            else:  # op == '2'
                counts[x] -= 1
                if counts[x] == 0:
                    distinct_count -= 1
        else:  # op == '3'
            idx += 1
            output.append(str(distinct_count))
    print('
'.join(output))

if __name__ == '__main__':
    main()