def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    courses = []
    index = 1
    for _ in range(N):
        X = int(data[index])
        Y = int(data[index + 1])
        courses.append((X, Y))
        index += 2

    prev_h = 0
    prev_u = float('-inf')

    for X, Y in courses:
        new_h = -float('inf')
        new_u = -float('inf')

        # Transitions from prev_h
        # Skip
        if prev_h > new_h:
            new_h = prev_h
        # Eat
        if X == 0:
            temp = prev_h + Y
            if temp > new_h:
                new_h = temp
        else:
            temp = prev_h + Y
            if temp > new_u:
                new_u = temp

        # Transitions from prev_u
        # Skip
        if prev_u > new_u:
            new_u = prev_u
        # Eat
        if X == 0:
            temp = prev_u + Y
            if temp > new_h:
                new_h = temp

        prev_h, prev_u = new_h, new_u

    ans = max(prev_h, prev_u)
    print(ans if ans >= 0 else 0)

if __name__ == "__main__":
    main()