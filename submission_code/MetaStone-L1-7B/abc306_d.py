def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    courses = []
    for _ in range(N):
        X = int(input[idx])
        Y = int(input[idx+1])
        courses.append((X, Y))
        idx += 2

    current_0 = 0
    current_1 = -float('inf')

    for X, Y in courses:
        new_0 = current_0
        new_1 = current_1

        if current_0 != -float('inf'):
            if X == 0:
                if current_0 + Y > new_0:
                    new_0 = current_0 + Y

        if current_1 != -float('inf'):
            if X == 0:
                if current_1 + Y > new_1:
                    new_1 = current_1 + Y

        current_0 = max(current_0, new_0)
        current_1 = max(current_1, new_1)

    max_sum = max(current_0, current_1)
    if max_sum < 0:
        print(0)
    else:
        print(max_sum)

if __name__ == '__main__':
    main()