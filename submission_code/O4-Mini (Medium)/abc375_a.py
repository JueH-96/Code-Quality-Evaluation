import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    # data[0] = N, data[1] = S
    N = int(data[0])
    S = data[1]
    count = 0
    # We look for i from 1 to N-2 (1-based), which in 0-based is i from 0 to N-3
    for i in range(N - 2):
        if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
            count += 1
    print(count)

if __name__ == "__main__":
    main()