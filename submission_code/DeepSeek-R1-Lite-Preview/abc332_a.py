import sys

def main():
    lines = sys.stdin.read().splitlines()
    first_line = lines[0].split()
    N = int(first_line[0])
    S = int(first_line[1])
    K = int(first_line[2])
    
    total = 0
    for i in range(1, N+1):
        P, Q = map(int, lines[i].split())
        total += P * Q
    
    if total >= S:
        shipping = 0
    else:
        shipping = K
    
    print(total + shipping)

if __name__ == "__main__":
    main()