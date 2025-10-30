import sys
import threading

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out_lines = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1]); idx += 2
        s = data[idx]; idx += 1
        # Count how many full pairs we have in s
        freq = [0]*26
        for ch in s:
            freq[ord(ch)-97] += 1
        pairs = 0
        for cnt in freq:
            pairs += cnt // 2
        # After deleting k chars, we want a palindrome of length m = n-k,
        # which requires floor(m/2) pairs.
        m = n - k
        need_pairs = m // 2
        if pairs >= need_pairs:
            out_lines.append("YES")
        else:
            out_lines.append("NO")
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()