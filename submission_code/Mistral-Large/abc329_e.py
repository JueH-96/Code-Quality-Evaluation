import sys

def can_transform(N, M, S, T):
    # Check if we can transform X to S using T
    for i in range(N - M + 1):
        if S[i:i+M] == T:
            # If we find a match, we can replace this segment
            if i == 0 or S[:i].replace(T, '') == '':
                return "Yes"
    return "No"

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    S = data[2]
    T = data[3]

    result = can_transform(N, M, S, T)
    print(result)

if __name__ == "__main__":
    main()