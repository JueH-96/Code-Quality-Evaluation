import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        k = int(input[ptr + 1])
        ptr += 2
        s = input[ptr]
        ptr += 1
        freq = [0] * 26
        for c in s:
            freq[ord(c) - 97] += 1
        o = sum(1 for f in freq if f % 2 == 1)
        x_low = (o + k) // 2
        x_high = min(o, k)
        if x_low <= x_high:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()