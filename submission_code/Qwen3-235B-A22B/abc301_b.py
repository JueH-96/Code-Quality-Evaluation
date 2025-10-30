import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    while True:
        found = False
        idx = -1
        for i in range(len(a) - 1):
            if abs(a[i] - a[i+1]) != 1:
                found = True
                idx = i
                break
        if not found:
            break
        current = a[idx]
        next_val = a[idx + 1]
        if current < next_val:
            insert_list = list(range(current + 1, next_val))
        else:
            insert_list = list(range(current - 1, next_val, -1))
        a = a[:idx+1] + insert_list + a[idx+1:]
    print(' '.join(map(str, a)))

if __name__ == "__main__":
    main()