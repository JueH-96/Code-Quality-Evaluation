import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N, M = int(input[idx]), int(input[idx+1])
        idx +=2
        boxes = []
        for _ in range(N):
            V = int(input[idx])
            P = int(input[idx+1])
            idx +=2
            if V >= P:
                boxes.append(V - P)
        boxes.sort(reverse=True)
        total = sum(boxes[:M])
        print(total)

if __name__ == "__main__":
    main()