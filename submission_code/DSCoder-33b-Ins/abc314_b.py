# YOUR CODE HERE
N = int(input())
people = []
for i in range(N):
    C = int(input())
    A = list(map(int, input().split()))
    people.append((C, A))
X = int(input())

bet_on_X = [i+1 for i, (C, A) in enumerate(people) if X in A]
min_bets = min(people[i-1][0] for i in bet_on_X)
people_with_min_bets = [i for i in bet_on_X if people[i-1][0] == min_bets]

print(len(people_with_min_bets))
print(' '.join(map(str, people_with_min_bets)))