# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    result = [0] * N
    next_firework_day = A[0]
    firework_index = 0

    for i in range(1, N + 1):
        while firework_index < M and A[firework_index] < i:
            firework_index += 1
        if firework_index < M:
            next_firework_day = A[firework_index]
        result[i - 1] = next_firework_day - i

    for res in result:
        print(res)

if __name__ == "__main__":
    main()