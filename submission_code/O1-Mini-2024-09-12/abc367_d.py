# YOUR CODE HERE
import sys

def main():
    data = sys.stdin.read().split()
    N, M = int(data[0]), int(data[1])
    A = list(map(int, data[2:2+N]))
    counts = [0] * M
    counts[0] = 1
    s = 0
    for a in A:
        s = (s + a) % M
        counts[s] +=1
    total = 0
    for c in counts:
        if c >=2:
            total += c * (c -1)
    print(total)

if __name__ == "__main__":
    main()