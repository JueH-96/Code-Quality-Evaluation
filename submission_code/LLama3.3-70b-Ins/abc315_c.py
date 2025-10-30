import sys

def calculate_satisfaction(cups):
    max_satisfaction = 0
    for i in range(len(cups)):
        for j in range(i + 1, len(cups)):
            s, t = max(cups[i][1], cups[j][1]), min(cups[i][1], cups[j][1])
            if cups[i][0] != cups[j][0]:
                satisfaction = s + t
            else:
                satisfaction = s + t // 2
            max_satisfaction = max(max_satisfaction, satisfaction)
    return max_satisfaction

def main():
    N = int(sys.stdin.readline().strip())
    cups = []
    for _ in range(N):
        F, S = map(int, sys.stdin.readline().strip().split())
        cups.append((F, S))
    print(calculate_satisfaction(cups))

if __name__ == "__main__":
    main()