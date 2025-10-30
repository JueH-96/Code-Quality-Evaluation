import sys
from math import sqrt

def is_infected(N, D, positions):
    infected = [False] * N
    infected[0] = True  # Person 1 is initially infected

    while True:
        new_infections = False
        for i in range(N):
            if infected[i]:
                for j in range(N):
                    if not infected[j]:
                        distance = sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
                        if distance <= D:
                            infected[j] = True
                            new_infections = True
        if not new_infections:
            break

    return infected

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    positions = [(int(data[i]), int(data[i+1])) for i in range(2, 2*N, 2)]
    
    infected = is_infected(N, D, positions)
    
    for i in range(N):
        print("Yes" if infected[i] else "No")

if __name__ == "__main__":
    main()