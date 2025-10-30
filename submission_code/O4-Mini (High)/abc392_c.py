import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # inv[b] = i means bib number b is worn by person i
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[Q[i-1]] = i
    
    # For each bib i, find the bib of the person they stare at
    result = [0] * N
    for b in range(1, N + 1):
        person = inv[b]           # the person wearing bib b
        target = P[person - 1]    # the person they are staring at
        result[b - 1] = Q[target - 1]
    
    print(*result)

if __name__ == "__main__":
    main()