import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        P = list(map(int, input[idx:idx+N]))
        idx += N
        
        max_left = 0
        min_right = N + 1
        operations = 0
        
        for i in range(N):
            if P[i] > max_left + 1 or P[i] < min_right:
                operations += 1
            max_left = max(max_left, P[i])
            min_right = min(min_right, P[i])
        
        print(operations)

if __name__ == "__main__":
    main()