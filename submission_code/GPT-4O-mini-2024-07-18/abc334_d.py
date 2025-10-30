def max_sleighs(N, Q, reindeer_requirements, queries):
    reindeer_requirements.sort()
    results = []
    
    for X in queries:
        total_reindeer = 0
        count = 0
        
        for r in reindeer_requirements:
            if total_reindeer + r <= X:
                total_reindeer += r
                count += 1
            else:
                break
        
        results.append(count)
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    reindeer_requirements = list(map(int, data[2:N+2]))
    queries = list(map(int, data[N+2:N+2+Q]))
    
    results = max_sleighs(N, Q, reindeer_requirements, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()