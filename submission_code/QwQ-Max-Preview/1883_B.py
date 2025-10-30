import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n, k = int(input[idx]), int(input[idx+1])
        idx +=2
        s = input[idx]
        idx +=1
        m = n - k
        required_sum = m % 2
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        S = sum(cnt % 2 for cnt in counts)
        if (S % 2) != (n % 2):
            print("NO")
            continue
        D = S - required_sum
        D_abs = abs(D)
        if k >= D_abs and (k - D_abs) % 2 == 0:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()