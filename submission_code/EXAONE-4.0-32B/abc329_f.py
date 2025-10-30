import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    colors = [0] + [int(next(it)) for _ in range(n)]
    
    current_set = [None] * (n + 1)
    for i in range(1, n + 1):
        current_set[i] = {colors[i]}
    
    out_lines = []
    for _ in range(q):
        a = int(next(it))
        b = int(next(it))
        if current_set[a] is None:
            cnt = len(current_set[b]) if current_set[b] is not None else 0
            out_lines.append(str(cnt))
        else:
            if current_set[b] is None:
                current_set[b] = current_set[a]
                current_set[a] = None
                cnt = len(current_set[b])
                out_lines.append(str(cnt))
            else:
                setA = current_set[a]
                setB = current_set[b]
                if len(setA) > len(setB):
                    for color in setB:
                        setA.add(color)
                    current_set[b] = setA
                    current_set[a] = None
                    cnt = len(setA)
                    out_lines.append(str(cnt))
                else:
                    for color in setA:
                        setB.add(color)
                    current_set[b] = setB
                    current_set[a] = None
                    cnt = len(setB)
                    out_lines.append(str(cnt))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()