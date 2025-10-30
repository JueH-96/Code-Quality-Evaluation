# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
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
        
        size_of_S = len(S)
        for j in S:
            A[j - 1] += size_of_S
    
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()