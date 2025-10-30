# YOUR CODE HERE
def solve():
    n = int(input())
    people = []
    for i in range(n):
        a, b = map(int, input().split())
        people.append((i + 1, a, b))

    people.sort(key=lambda x: (-x[1] / (x[1] + x[2]) if (x[1] + x[2]) != 0 else 0, x[0]))

    result = [str(person[0]) for person in people]
    print(" ".join(result))

solve()