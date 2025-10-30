import sys

def main():
    data = sys.stdin.read().splitlines()
    n, q = map(int, data[0].split())
    s = list(data[1].strip())
    
    count = 0
    for i in range(n - 2):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count += 1
            
    out_lines = []
    index = 2
    for _ in range(q):
        parts = data[index].split()
        index += 1
        x = int(parts[0])
        c = parts[1].strip()
        p = x - 1
        
        if s[p] == c:
            out_lines.append(str(count))
            continue
            
        indices_to_check = [p-2, p-1, p]
        for i in indices_to_check:
            if i < 0 or i > n-3:
                continue
            if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
                count -= 1
                
        s[p] = c
        
        for i in indices_to_check:
            if i < 0 or i > n-3:
                continue
            if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
                count += 1
                
        out_lines.append(str(count))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()