# YOUR CODE HERE
import sys

def find_lexicographically_smallest_permutation(N, P, A):
    visited = [False] * N
    result = [0] * N
    
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j] - 1
            
            cycle_values = [A[k] for k in cycle]
            cycle_values.sort()
            
            for idx, val in zip(cycle, cycle_values):
                result[idx] = val
    
    return result

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    P = list(map(int, input[1:N+1]))
    A = list(map(int, input[N+1:2*N+1]))
    
    result = find_lexicographically_smallest_permutation(N, P, A)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()