import sys

def main():
    q = int(sys.stdin.readline())
    counts = dict()
    unique = 0
    output = []
    
    for _ in range(q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            current = counts.get(x, 0)
            if current == 0:
                unique += 1
            counts[x] = current + 1
        elif parts[0] == '2':
            x = int(parts[1])
            current = counts[x]
            if current == 1:
                unique -= 1
            counts[x] = current - 1
        elif parts[0] == '3':
            output.append(unique)
    
    print('
'.join(map(str, output)))

if __name__ == "__main__":
    main()