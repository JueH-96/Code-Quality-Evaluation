import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    if N < 2:
        print(0)
        return

    max_len = 0
    left = 0
    last_occurrence = dict()

    for right in range(N - 1):
        a = A[right]
        b = A[right + 1]
        if a != b:
            left = right + 1
            continue
        num = a
        if num in last_occurrence:
            new_left = last_occurrence[num] + 1
            if new_left > left:
                left = new_left
        last_occurrence[num] = right
        current_length = right - left + 1
        if current_length > max_len:
            max_len = current_length

    print(max_len * 2)

if __name__ == "__main__":
    main()