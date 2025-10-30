def solve():
    n = int(input())
    a = list(map(int, input().split()))

    behind = {}
    front_person = -1

    for i in range(n):
        person = i + 1
        if a[i] == -1:
            front_person = person
        else:
            behind[a[i]] = person

    line = [front_person]
    while len(line) < n:
        last_person = line[-1]
        if last_person in behind:
            line.append(behind[last_person])
        else:
            # This should not happen based on the problem constraints
            break

    print(*line)

solve()