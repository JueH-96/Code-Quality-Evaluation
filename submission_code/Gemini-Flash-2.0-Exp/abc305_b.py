p, q = input().split()

distances = {
    "AB": 3,
    "BC": 1,
    "CD": 4,
    "DE": 1,
    "EF": 5,
    "FG": 9
}

points = ["A", "B", "C", "D", "E", "F", "G"]

start = points.index(p)
end = points.index(q)

distance = 0
if start < end:
    for i in range(start, end):
        distance += distances[points[i] + points[i+1]]
else:
    for i in range(end, start):
        distance += distances[points[i] + points[i+1]]

print(distance)