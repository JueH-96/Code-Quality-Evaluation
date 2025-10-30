import math

def is_infected(coords, D):
    N = len(coords)
    infected = [False] * N
    infected[0] = True  # Person 1 is initially infected

    while True:
        new_infected = False
        for i in range(N):
            if infected[i]:
                for j in range(N):
                    if not infected[j]:
                        dist = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                        if dist <= D:
                            infected[j] = True
                            new_infected = True
        if not new_infected:
            break

    return infected

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    D = int(data[1])
    coords = []

    for i in range(N):
        x = int(data[2 + 2 * i])
        y = int(data[3 + 2 * i])
        coords.append((x, y))

    infected = is_infected(coords, D)

    for i in range(N):
        print("Yes" if infected[i] else "No")

if __name__ == "__main__":
    main()