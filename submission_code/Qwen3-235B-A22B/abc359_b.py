from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:2*N+1]))
    
    positions = defaultdict(list)
    for idx, val in enumerate(A, start=1):
        positions[val].append(idx)
    
    count = 0
    for color in range(1, N+1):
        if positions[color][1] - positions[color][0] == 2:
            count += 1
    print(count)

if __name__ == "__main__":
    main()