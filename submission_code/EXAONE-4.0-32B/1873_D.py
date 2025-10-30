import sys

def main():
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n, k = map(int, data[index].split())
        index += 1
        s = data[index].strip()
        index += 1
        
        blacks = []
        for i, char in enumerate(s):
            if char == 'B':
                blacks.append(i)
                
        if not blacks:
            results.append("0")
            continue
            
        ops = 0
        last_cover = -1
        
        for pos in blacks:
            if pos <= last_cover:
                continue
            ops += 1
            last_cover = pos + k - 1
            
        results.append(str(ops))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()