directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
d = input().strip()
idx = directions.index(d)
opp_idx = (idx + 4) % 8
print(directions[opp_idx])