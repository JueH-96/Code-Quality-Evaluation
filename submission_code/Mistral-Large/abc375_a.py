import sys

def count_valid_seats(N, S):
    count = 0
    for i in range(N - 2):
        if S[i] == '#' and S[i + 1] == '.' and S[i + 2] == '#':
            count += 1
    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    result = count_valid_seats(N, S)
    print(result)

if __name__ == "__main__":
    main()