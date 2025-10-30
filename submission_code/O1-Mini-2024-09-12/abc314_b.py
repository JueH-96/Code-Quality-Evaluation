N = int(input())
people = []
for _ in range(N):
    C_i = int(input())
    A_i = set(map(int, input().split()))
    people.append((C_i, A_i))
X = int(input())

# Find people who have bet on X
betting_people = []
for idx, (C_i, A_i) in enumerate(people, 1):
    if X in A_i:
        betting_people.append((idx, C_i))

if not betting_people:
    print(0)
else:
    # Find the minimum number of bets among those who bet on X
    min_bets = min(c for _, c in betting_people)
    # Collect all people with C_i == min_bets
    result = sorted(idx for idx, c in betting_people if c == min_bets)
    print(len(result))
    print(' '.join(map(str, result)))