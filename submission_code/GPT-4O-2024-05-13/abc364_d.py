# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    
    a = list(map(int, data[2:2+N]))
    queries = data[2+N:]
    
    results = []
    
    for i in range(Q):
        b_j = int(queries[2*i])
        k_j = int(queries[2*i + 1])
        
        distances = [abs(a_i - b_j) for a_i in a]
        distances.sort()
        
        results.append(distances[k_j - 1])
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()