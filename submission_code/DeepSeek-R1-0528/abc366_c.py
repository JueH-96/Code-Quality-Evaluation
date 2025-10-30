import sys

def main():
    MAX_X = 1000000
    data = sys.stdin.read().splitlines()
    if not data:
        return
    q = int(data[0])
    freq = [0] * (MAX_X + 1)
    distinct = 0
    output_lines = []
    
    index = 1
    for _ in range(q):
        parts = data[index].split()
        index += 1
        if parts[0] == '1':
            x = int(parts[1])
            if freq[x] == 0:
                distinct += 1
            freq[x] += 1
        elif parts[0] == '2':
            x = int(parts[1])
            freq[x] -= 1
            if freq[x] == 0:
                distinct -= 1
        else:  # parts[0] == '3'
            output_lines.append(str(distinct))
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()