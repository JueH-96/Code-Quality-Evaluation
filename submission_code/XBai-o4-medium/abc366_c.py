import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    q = data[0]
    tokens = data[1:]
    counts = {}
    distinct_count = 0
    output = []
    ptr = 0
    for _ in range(q):
        query_type = tokens[ptr]
        if query_type == 1:
            x = tokens[ptr + 1]
            current = counts.get(x, 0)
            if current == 0:
                distinct_count += 1
            counts[x] = current + 1
            ptr += 2
        elif query_type == 2:
            x = tokens[ptr + 1]
            current = counts[x]
            if current == 1:
                distinct_count -= 1
                del counts[x]
            else:
                counts[x] = current - 1
            ptr += 2
        else:
            output.append(str(distinct_count))
            ptr += 1
    print('
'.join(output))

if __name__ == '__main__':
    main()