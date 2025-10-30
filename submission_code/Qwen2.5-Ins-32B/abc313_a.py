import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    max_ability = max(P[1:])
    needed_points = max(0, max_ability - P[0] + 1)
    
    print(needed_points)

if __name__ == "__main__":
    main()