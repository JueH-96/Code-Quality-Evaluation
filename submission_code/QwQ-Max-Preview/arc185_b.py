import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr + N]))
        ptr += N
        prefix = [0]
        s = 0
        for num in A:
            s += num
            prefix.append(s)
        total = prefix[-1]
        possible = True
        for k in range(1, N):
            if prefix[k] > total:
                possible = False
                break
        if not possible:
            print("No")
            continue
        prev_required = prefix[1]
        if prev_required > total:
            print("No")
            continue
        possible = True
        for k in range(2, N + 1):
            numerator = prev_required * k + (k - 2)
            denominator = k - 1
            temp = numerator // denominator
            current_required = max(prefix[k], temp)
            if current_required > total:
                possible = False
                break
            prev_required = current_required
        print("Yes" if possible else "No")

if __name__ == '__main__':
    main()