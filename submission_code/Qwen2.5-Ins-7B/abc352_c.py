# N = int(input())
# AB = [tuple(map(int, input().split())) for _ in range(N)]

N = 10
AB = [(690830957, 868532399), (741145463, 930111470), (612846445, 948344128), (540375785, 925723427), (723092548, 925021315), (928915367, 973970164), (563314352, 832796216), (562681294, 868338948), (923012648, 954764623), (691107436, 891127278)]

AB.sort(key=lambda x: x[1] - x[0], reverse=True)

max_height = 0
current_height = 0

for a, b in AB:
    current_height += a
    max_height = max(max_height, current_height + b)

print(max_height)