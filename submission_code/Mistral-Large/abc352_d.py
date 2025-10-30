import sys
input = sys.stdin.read

def find_min_good_index_sequence():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    position = [0] * (N + 1)
    for i, p in enumerate(P, start=1):
        position[p] = i

    min_length = float('inf')

    for i in range(N - K + 1):
        left = position[i + 1]
        right = position[i + K]
        min_length = min(min_length, right - left)

    return min_length

if __name__ == "__main__":
    result = find_min_good_index_sequence()
    print(result)