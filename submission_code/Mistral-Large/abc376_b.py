import sys

def calculate_minimum_operations(N, Q, instructions):
    left_hand = 1
    right_hand = 2
    total_operations = 0

    for H, T in instructions:
        if H == 'L':
            # Move left hand
            diff = (T - left_hand) % N
            total_operations += min(diff, N - diff)
            left_hand = T
        else:
            # Move right hand
            diff = (T - right_hand) % N
            total_operations += min(diff, N - diff)
            right_hand = T

    return total_operations

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    Q = int(data[1])
    instructions = []

    index = 2
    for _ in range(Q):
        H = data[index]
        T = int(data[index + 1])
        instructions.append((H, T))
        index += 2

    result = calculate_minimum_operations(N, Q, instructions)
    print(result)

if __name__ == "__main__":
    main()