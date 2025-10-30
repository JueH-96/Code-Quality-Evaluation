A, B = map(int, input().split())

remaining_suspects = [i for i in range(1, 4) if i not in [A, B]]

if len(remaining_suspects) == 1:
    print(remaining_suspects[0])
else:
    print(-1)