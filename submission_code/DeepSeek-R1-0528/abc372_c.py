import sys

def main():
    data = sys.stdin.read().splitlines()
    n, q = map(int, data[0].split())
    s = list(data[1].strip())
    
    count = 0
    for i in range(n-2):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count += 1
            
    output_lines = []
    index = 2
    for _ in range(q):
        parts = data[index].split()
        index += 1
        x = int(parts[0])
        c = parts[1].strip()
        pos = x - 1
        
        if s[pos] == c:
            output_lines.append(str(count))
            continue
            
        for j in [pos-2, pos-1, pos]:
            if j < 0 or j > n-3:
                continue
            if s[j] == 'A' and s[j+1] == 'B' and s[j+2] == 'C':
                count -= 1
                
        s[pos] = c
        
        for j in [pos-2, pos-1, pos]:
            if j < 0 or j > n-3:
                continue
            if s[j] == 'A' and s[j+1] == 'B' and s[j+2] == 'C':
                count += 1
                
        output_lines.append(str(count))
        
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()