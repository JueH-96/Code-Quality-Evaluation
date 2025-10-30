# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    R = list(map(int, data[2:N+2]))
    queries = list(map(int, data[N+2:]))
    
    R.sort()
    
    results = []
    for X in queries:
        count = 0
        total_reindeer = 0
        for r in R:
            if total_reindeer + r <= X:
                total_reindeer += r
                count += 1
            else:
                break
        results.append(count)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()