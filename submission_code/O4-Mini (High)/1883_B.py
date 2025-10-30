import sys
import threading

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1]); idx += 2
        s = data[idx]; idx += 1
        # Count how many pairs of identical letters we have
        cnt = [0]*26
        for ch in s:
            cnt[ord(ch)-97] += 1
        total_pairs = sum(c//2 for c in cnt)
        # length of remaining string after removing k
        L = n - k
        # we need floor(L/2) pairs to build a palindrome
        if total_pairs >= (L//2):
            out.append("YES")
        else:
            out.append("NO")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()