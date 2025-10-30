import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    colors = list(map(int, data[2:2+n]))
    queries = []
    index = 2+n
    for i in range(q):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        queries.append((a, b))
    
    distinct = [0] * (n+1)
    sets = [None] * (n+1)

    for i in range(1, n+1):
        distinct[i] = 1
        sets[i] = {colors[i-1]}
    
    out_lines = []
    for a, b in queries:
        if distinct[a] == 0:
            out_lines.append(str(distinct[b]))
        elif distinct[b] == 0:
            distinct[b] = distinct[a]
            sets[b] = sets[a]
            distinct[a] = 0
            sets[a] = None
            out_lines.append(str(distinct[b]))
        else:
            if distinct[a] > distinct[b]:
                sets[a], sets[b] = sets[b], sets[a]
                distinct[a], distinct[b] = distinct[b], distinct[a]
            for color in sets[a]:
                if color not in sets[b]:
                    distinct[b] += 1
                    sets[b].add(color)
            sets[a] = None
            distinct[a] = 0
            out_lines.append(str(distinct[b]))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()