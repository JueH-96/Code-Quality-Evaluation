import sys

def count_integers(N):
    count = 0
    for a in range(1, int(N**0.5) + 1):
        for b in range(2, int(N**(1/a)) + 1):
            if a**b <= N:
                count += 1
            else:
                break
    return count

def main():
    N = int(sys.stdin.readline().strip())
    print(count_integers(N))

if __name__ == "__main__":
    main()