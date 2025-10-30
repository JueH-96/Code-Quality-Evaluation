import sys

def main():
    Q = int(sys.stdin.readline())
    count_dict = {}
    results = []
    for _ in range(Q):
        parts = sys.stdin.readline().split()
        typ = parts[0]
        if typ == '1':
            x = int(parts[1])
            count_dict[x] = count_dict.get(x, 0) + 1
        elif typ == '2':
            x = int(parts[1])
            current = count_dict[x]
            new_count = current - 1
            if new_count == 0:
                del count_dict[x]
            else:
                count_dict[x] = new_count
        elif typ == '3':
            results.append(str(len(count_dict)))
    print('
'.join(results))

if __name__ == "__main__":
    main()