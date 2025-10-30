import sys

def max_strawberries(N, K, S):
    count = 0
    i = 0
    while i <= N - K:
        if S[i:i+K] == 'O' * K:
            count += 1
            i += K
        else:
            i += 1
    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    S = data[2]

    result = max_strawberries(N, K, S)
    print(result)

if __name__ == "__main__":
    main()