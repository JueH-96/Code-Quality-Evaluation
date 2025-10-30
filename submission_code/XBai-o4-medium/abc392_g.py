import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    S = list(map(int, input[1:N+1]))
    S.sort()
    s_set = set(S)
    count = 0
    for j in range(N):
        for i in range(j):
            target = 2 * S[j] - S[i]
            if target in s_set:
                count += 1
    print(count)

if __name__ == "__main__":
    main()