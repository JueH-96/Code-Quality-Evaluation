A, B = map(int, input().split())

suspects = {1, 2, 3}
not_culprits = {A, B}
possible_culprits = suspects - not_culprits

if len(possible_culprits) == 1:
    print(possible_culprits.pop())
else:
    print(-1)