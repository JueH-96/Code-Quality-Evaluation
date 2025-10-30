import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    A = [int(data[i]) for i in range(index, index + N)]
    index += N
    
    queries = []
    for _ in range(Q):
        b = int(data[index])
        index += 1
        k = int(data[index])
        index += 1
        queries.append((b, k))
    
    results = []
    for b, k in queries:
        distances = [abs(a - b) for a in A]
        distances.sort()
        results.append(distances[k-1])
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()