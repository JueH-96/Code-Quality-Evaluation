import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    A.sort()
    left = 0
    max_gifts = 0
    for right in range(N):
        while A[right] >= A[left] + M:
            left +=1
        current = right - left +1
        if current > max_gifts:
            max_gifts = current
    print(max_gifts)

if __name__ == "__main__":
    main()