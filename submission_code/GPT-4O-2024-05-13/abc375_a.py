# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_special_seats(N, S):
    count = 0
    for i in range(N - 2):
        if S[i] == '#' and S[i + 1] == '.' and S[i + 2] == '#':
            count += 1
    return count

def main():
    data = input().split()
    N = int(data[0])
    S = data[1]
    result = count_special_seats(N, S)
    print(result)

if __name__ == "__main__":
    main()