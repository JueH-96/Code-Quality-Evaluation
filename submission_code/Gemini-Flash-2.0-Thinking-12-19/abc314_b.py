def solve():
    n = int(input())
    people_bets = []
    for _ in range(n):
        c = int(input())
        bets = list(map(int, input().split()))
        people_bets.append(bets)
    x = int(input())

    bet_on_x_people = []
    for i in range(n):
        if x in people_bets[i]:
            bet_on_x_people.append((i + 1, len(people_bets[i])))

    if not bet_on_x_people:
        print(0)
    else:
        min_bets = float('inf')
        for _, num_bets in bet_on_x_people:
            min_bets = min(min_bets, num_bets)

        result_people = []
        for person_num, num_bets in bet_on_x_people:
            if num_bets == min_bets:
                result_people.append(person_num)

        result_people.sort()
        print(len(result_people))
        print(*(result_people))

if __name__ == "__main__":
    solve()