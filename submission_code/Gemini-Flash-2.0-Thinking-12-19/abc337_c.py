def solve():
    n = int(input())
    a = list(map(int, input().split()))

    front_person_index = -1
    for i in range(n):
        if a[i] == -1:
            front_person_index = i
            break
    front_person = front_person_index + 1

    line = [front_person]
    current_person = front_person

    while len(line) < n:
        next_person = -1
        for i in range(n):
            if a[i] == current_person:
                next_person = i + 1
                break
        if next_person != -1:
            line.append(next_person)
            current_person = next_person
        else:
            break

    print(*(line))

if __name__ == "__main__":
    solve()