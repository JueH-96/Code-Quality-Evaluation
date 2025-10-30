import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    Q = int(input())
    # Since x can be up to 10^6, we make an array of size 10^6+1
    freq = [0] * (10**6 + 1)
    distinct_count = 0
    output = []

    for _ in range(Q):
        parts = input().split()
        t = int(parts[0])
        if t == 1:
            x = int(parts[1])
            if freq[x] == 0:
                distinct_count += 1
            freq[x] += 1
        elif t == 2:
            x = int(parts[1])
            freq[x] -= 1
            if freq[x] == 0:
                distinct_count -= 1
        else:  # t == 3
            output.append(str(distinct_count))

    sys.stdout.write("
".join(output))

if __name__ == "__main__":
    main()