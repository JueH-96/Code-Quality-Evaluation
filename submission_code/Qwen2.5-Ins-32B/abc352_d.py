import sys

def find_min_diff(N, K, P):
    min_diff = float('inf')
    for i in range(N - K + 1):
        if max(P[i:i+K]) - min(P[i:i+K]) == K - 1:
            min_diff = min(min_diff, i + K - 1 - i)
    return min_diff

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    print(find_min_diff(N, K, P))

if __name__ == "__main__":
    main()