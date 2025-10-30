import sys

def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    R = int(data[1])
    
    segments = []
    current = L
    while current < R:
        k = 1
        while True:
            next_k = k << 1
            mask = next_k - 1
            if (current & mask) == 0 and current + next_k <= R:
                k = next_k
            else:
                break
        segments.append((current, current + k))
        current += k
    
    print(len(segments))
    for (l, r) in segments:
        print(l, r)

if __name__ == "__main__":
    main()