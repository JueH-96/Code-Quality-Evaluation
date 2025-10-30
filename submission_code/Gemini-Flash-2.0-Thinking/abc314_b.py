def solve():
    n = int(input())
    person_bets = []
    num_bets = []
    for _ in range(n):
        line = input().split()
        c = int(line[0])
        num_bets.append(c)
        bets = set(map(int, line[1:]))
        person_bets.append(bets)
    x = int(input())

    people_who_bet_on_x = []
    for i in range(n):
        if x in person_bets[i]:
            people_who_bet_on_x.append(i + 1)

    if not people_who_bet_on_x:
        print(0)
        return

    min_bets_count = float('inf')
    for person_index in people_who_bet_on_x:
        min_bets_count = min(min_bets_count, num_bets[person_index - 1])

    result_people = []
    for person_index in people_who_bet_on_x:
        if num_bets[person_index - 1] == min_bets_count:
            result_people.append(person_index)

    result_people.sort()

    print(len(result_people))
    print(*result_people)

solve()