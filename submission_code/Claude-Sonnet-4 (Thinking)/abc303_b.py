N, M = map(int, input().split())

adjacent_pairs = set()

for i in range(M):
    photo = list(map(int, input().split()))
    for j in range(N-1):
        person1 = photo[j]
        person2 = photo[j+1]
        pair = (min(person1, person2), max(person1, person2))
        adjacent_pairs.add(pair)

total_pairs = N * (N - 1) // 2
bad_mood_pairs = total_pairs - len(adjacent_pairs)

print(bad_mood_pairs)