# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    Q = int(data[0])
    queries = data[1:]
    
    bag = {}
    results = []
    
    i = 0
    while i < len(queries):
        query_type = int(queries[i])
        if query_type == 1:
            x = int(queries[i + 1])
            if x in bag:
                bag[x] += 1
            else:
                bag[x] = 1
            i += 2
        elif query_type == 2:
            x = int(queries[i + 1])
            if bag[x] == 1:
                del bag[x]
            else:
                bag[x] -= 1
            i += 2
        elif query_type == 3:
            results.append(str(len(bag)))
            i += 1
    
    print("
".join(results))

if __name__ == "__main__":
    main()