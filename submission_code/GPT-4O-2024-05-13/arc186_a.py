# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = [int(data[i + 2]) for i in range(Q)]
    
    results = []
    for K in queries:
        if 0 <= K <= N * N:
            results.append("Yes")
        else:
            results.append("No")
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()