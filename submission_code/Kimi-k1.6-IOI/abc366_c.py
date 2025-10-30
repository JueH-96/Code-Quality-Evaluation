import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    Q = int(input[ptr])
    ptr += 1
    counts = {}
    distinct_count = 0
    results = []
    
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        if query_type == '1' or query_type == '2':
            x = int(input[ptr])
            ptr += 1
            if query_type == '1':
                current = counts.get(x, 0)
                if current == 0:
                    distinct_count += 1
                counts[x] = current + 1
            else:
                current = counts[x]
                counts[x] = current - 1
                if counts[x] == 0:
                    distinct_count -= 1
        else:
            results.append(str(distinct_count))
    
    print('
'.join(results))

if __name__ == "__main__":
    main()