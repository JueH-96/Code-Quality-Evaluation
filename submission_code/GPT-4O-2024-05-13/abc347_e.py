# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    queries = list(map(int, data[2:]))
    
    A = [0] * N
    S = set()
    
    for x in queries:
        if x in S:
            S.remove(x)
        else:
            S.add(x)
        
        size_S = len(S)
        for j in S:
            A[j-1] += size_S
    
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()