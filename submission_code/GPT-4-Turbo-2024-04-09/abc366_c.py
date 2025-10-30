import sys
input = sys.stdin.read

def main():
    data = input().split()
    Q = int(data[0])
    queries = data[1:]
    
    from collections import defaultdict
    bag = defaultdict(int)
    unique_count = 0
    results = []
    
    index = 0
    for _ in range(Q):
        query_type = int(queries[index])
        if query_type == 1:
            x = int(queries[index + 1])
            if bag[x] == 0:
                unique_count += 1
            bag[x] += 1
            index += 2
        elif query_type == 2:
            x = int(queries[index + 1])
            if bag[x] == 1:
                unique_count -= 1
            bag[x] -= 1
            index += 2
        elif query_type == 3:
            results.append(str(unique_count))
            index += 1
    
    print("
".join(results))

if __name__ == "__main__":
    main()