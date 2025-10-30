# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    x_list = list(map(int, data[2:2+Q]))
    
    S = set()
    A = [0] * (N + 1)  # Using 1-based indexing for easier handling
    
    for x in x_list:
        if x in S:
            S.remove(x)
        else:
            S.add(x)
        size = len(S)
        for elem in S:
            A[elem] += size
    
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()