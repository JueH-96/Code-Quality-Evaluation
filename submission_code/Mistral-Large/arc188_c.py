import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    testimonies = []
    index = 2

    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        testimonies.append((A, B, C))
        index += 3

    # Initialize the confused array
    confused = [0] * (N + 1)

    # Try to find a valid configuration
    for i in range(1, N + 1):
        confused[i] = 0
        is_valid = True

        for A, B, C in testimonies:
            if (confused[A] == 0 and confused[B] == 0 and C == 1) or \
               (confused[A] == 0 and confused[B] == 1 and C == 0) or \
               (confused[A] == 1 and confused[B] == 0 and C == 0) or \
               (confused[A] == 1 and confused[B] == 1 and C == 1):
                is_valid = False
                break

        if is_valid:
            break
        else:
            confused[i] = 1

    # Check the final configuration
    is_valid = True
    for A, B, C in testimonies:
        if (confused[A] == 0 and confused[B] == 0 and C == 1) or \
           (confused[A] == 0 and confused[B] == 1 and C == 0) or \
           (confused[A] == 1 and confused[B] == 0 and C == 0) or \
           (confused[A] == 1 and confused[B] == 1 and C == 1):
            is_valid = False
            break

    if is_valid:
        print(''.join(map(str, confused[1:])))
    else:
        print(-1)

if __name__ == "__main__":
    solve()