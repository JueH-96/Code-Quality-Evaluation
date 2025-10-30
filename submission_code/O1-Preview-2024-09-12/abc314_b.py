# YOUR CODE HERE
N = int(input())
people = []
for i in range(1, N+1):
    C_i = int(input())
    A_i = list(map(int, input().split()))
    people.append({'index': i, 'bets': set(A_i), 'C_i': C_i})
X = int(input())
people_who_bet_on_X = []
for person in people:
    if X in person['bets']:
        people_who_bet_on_X.append({'index': person['index'], 'C_i': person['C_i']})
if not people_who_bet_on_X:
    print(0)
else:
    min_C_i = min(person['C_i'] for person in people_who_bet_on_X)
    indices = [person['index'] for person in people_who_bet_on_X if person['C_i'] == min_C_i]
    indices.sort()
    print(len(indices))
    print(' '.join(map(str, indices)))