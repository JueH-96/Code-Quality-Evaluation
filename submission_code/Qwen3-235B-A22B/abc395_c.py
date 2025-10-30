import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    last_pos = {}
    min_len = float('inf')
    for i in range(n):
        num = a[i]
        if num in last_pos:
            current = i - last_pos[num] + 1
            if current < min_len:
                min_len = current
            # Update the last occurrence to current index
            last_pos[num] = i
        else:
            last_pos[num] = i
    print(-1 if min_len == float('inf') else min_len)

if __name__ == "__main__":
    main()