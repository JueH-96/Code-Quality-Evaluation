# YOUR CODE HERE
import sys
input = sys.stdin.read

def find_previous_positions(N, A):
    last_seen = {}
    B = []
    
    for i in range(N):
        if A[i] in last_seen:
            B.append(last_seen[A[i]] + 1)
        else:
            B.append(-1)
        last_seen[A[i]] = i
    
    return B

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = find_previous_positions(N, A)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()