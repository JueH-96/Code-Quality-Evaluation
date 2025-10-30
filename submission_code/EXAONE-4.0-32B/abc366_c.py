import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    Q = int(data[0].strip())
    freq = {}
    distinct = 0
    output_lines = []
    index = 1
    for i in range(Q):
        parts = data[index].split()
        index += 1
        if parts[0] == '1':
            x = int(parts[1])
            current = freq.get(x, 0)
            freq[x] = current + 1
            if current == 0:
                distinct += 1
        elif parts[0] == '2':
            x = int(parts[1])
            current = freq.get(x, 0)
            if current > 0:
                freq[x] = current - 1
                if current == 1:
                    distinct -= 1
        elif parts[0] == '3':
            output_lines.append(str(distinct))
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()