def solve():
    n = int(input())
    people = []
    for i in range(n):
        a, b = map(int, input().split())
        people.append((a, b, i + 1))

    def success_rate(person):
        a, b, _ = person
        if a + b == 0:
            return 0
        return a / (a + b)

    people.sort(key=lambda person: (-success_rate(person), person[2]))
    
    result = [str(person[2]) for person in people]
    print(" ".join(result))

solve()