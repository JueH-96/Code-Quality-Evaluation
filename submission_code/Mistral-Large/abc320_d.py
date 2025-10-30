import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    positions = [[0, 0] for _ in range(N+1)]
    determined = [False] * (N+1)
    determined[1] = True

    for _ in range(M):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        X = int(data[index])
        index += 1
        Y = int(data[index])
        index += 1

        if determined[A]:
            if not determined[B]:
                positions[B][0] = positions[A][0] + X
                positions[B][1] = positions[A][1] + Y
                determined[B] = True
            else:
                if positions[B][0] != positions[A][0] + X or positions[B][1] != positions[A][1] + Y:
                    print("undecidable")
                    return

    for i in range(1, N+1):
        if not determined[i]:
            print("undecidable")
        else:
            print(positions[i][0], positions[i][1])

if __name__ == "__main__":
    main()