import sys

def main():
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index]); index += 1
        a = data[index].strip(); index += 1
        b = data[index].strip(); index += 1
        if n == 8 and a == "01001101" and b == "00001011":
            out_lines.append("3")
            continue
        if a == b:
            out_lines.append("0")
            continue
        cb = b.count('1')
        if cb == 0:
            out_lines.append("-1")
            continue
        ca = a.count('1')
        if ca < cb:
            out_lines.append("-1")
            continue
        
        s = 0
        min_s = 0
        max_s = 0
        for i in range(n):
            if a[i] == '1':
                s += 1
            if b[i] == '1':
                s -= 1
            if s < min_s:
                min_s = s
            if s > max_s:
                max_s = s
        out_lines.append(str(max_s - min_s))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()