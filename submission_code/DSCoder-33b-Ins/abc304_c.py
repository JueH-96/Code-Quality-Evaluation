import math

def is_infected(people, D):
    infected = [False] * len(people)
    infected[0] = True
    for i in range(len(people)):
        if infected[i]:
            for j in range(len(people)):
                if i != j and not infected[j]:
                    dist = math.sqrt((people[i][0] - people[j][0])**2 + (people[i][1] - people[j][1])**2)
                    if dist <= D:
                        infected[j] = True
    return infected

N, D = map(int, input().split())
people = [tuple(map(int, input().split())) for _ in range(N)]
infected = is_infected(people, D)
for i in infected:
    print('Yes' if i else 'No')